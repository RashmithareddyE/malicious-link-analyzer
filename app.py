"""
app.py - Streamlit dashboard for the Malicious Link Analyzer.

Run from the project root:   streamlit run app.py

Security: URLs decoded from QR codes are untrusted text. They are only displayed
inside st.code() blocks or HTML-escaped; they are NEVER opened, fetched or downloaded.
"""
import html
import inspect
import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for servers
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from modules import history_manager, ml_classifier, report_generator
from modules.qr_scanner import (ImageValidationError, PYZBAR_AVAILABLE,
                                scan_qr_image, validate_image_bytes)
from modules.rag_engine import RAGEngine, RAGError
from modules.risk_assessment import assess
from modules.url_analyzer import analyze_url, validate_url

st.set_page_config(page_title=config.APP_TITLE, page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

# UI palette (charcoal / teal / cyan / purple). Risk colours stay semantic
# (green / orange / red) and come from config.COLORS.
UI_TEXT = "#e6e8ec"
UI_MUTED = "#9aa1ad"
UI_BG = "#121316"
UI_PANEL = "#1c1f26"
UI_PANEL2 = "#232730"
UI_LINE = "#2e333d"
UI_TEAL = "#14b8a6"
UI_CYAN = "#22d3ee"
UI_PURPLE = "#8b5cf6"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def stretch(fn) -> dict:
    """Full-width kwargs that work on both old and new Streamlit versions."""
    params = inspect.signature(fn).parameters
    return {"width": "stretch"} if "width" in params else {"use_container_width": True}


def load_css() -> None:
    try:
        css = config.STYLE_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except OSError:
        st.warning("assets/style.css not found - using the default theme.")


def card(label: str, value, css: str = "c-cyan", sub: str = "") -> str:
    """HTML for a dashboard card. All dynamic text is escaped."""
    return (f"<div class='card'><div class='label'>{html.escape(str(label))}</div>"
            f"<div class='value {css}'>{html.escape(str(value))}</div>"
            f"<div class='sub'>{html.escape(sub)}</div></div>")


def risk_css(category: str) -> str:
    return {"LOW RISK": "low", "MEDIUM RISK": "medium", "HIGH RISK": "high"}.get(category, "medium")


def risk_text_css(category: str) -> str:
    return {"LOW RISK": "c-green", "MEDIUM RISK": "c-orange", "HIGH RISK": "c-red"}.get(category, "c-cyan")


@st.cache_resource(show_spinner="Loading machine learning model...")
def get_model():
    """Load (or auto-train on the demo dataset) the classifier."""
    try:
        return ml_classifier.ensure_model(), None
    except Exception as exc:  # missing dataset, corrupt model, ...
        return None, str(exc)


@st.cache_resource(show_spinner="Loading knowledge base and FAISS index...")
def get_rag():
    try:
        engine = RAGEngine()
        engine.load()
        return engine, None
    except Exception as exc:
        return None, str(exc)


# --------------------------------------------------------------------------
# Core analysis pipeline
# --------------------------------------------------------------------------
def run_analysis(raw_text: str, source: str):
    """Validate -> rules -> ML -> final score -> RAG explanation -> history."""
    ok, normalized, message = validate_url(raw_text)
    if not ok:
        return None, message
    rule = analyze_url(raw_text.strip())

    bundle, model_err = get_model()
    ml_result = ml_classifier.predict_url(normalized, bundle) if bundle else None

    assessment = assess(rule, ml_result)

    engine, rag_err = get_rag()
    explanation = None
    if engine:
        try:
            explanation = engine.explain_url(assessment.indicators)
        except RAGError as exc:
            rag_err = str(exc)

    history_manager.add_record(assessment, source=source)
    result = {"assessment": assessment, "rule": rule, "explanation": explanation,
              "model_error": model_err, "rag_error": rag_err, "message": message}
    st.session_state["latest"] = result
    return result, message


# --------------------------------------------------------------------------
# Result rendering (shared by the scanner and URL analysis pages)
# --------------------------------------------------------------------------
def render_gauge(score: int, category: str) -> go.Figure:
    color = config.COLORS[category]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"color": color}},
        gauge={"axis": {"range": [0, 100], "tickcolor": UI_MUTED},
               "bar": {"color": color, "thickness": 0.28},
               "bgcolor": UI_PANEL2,
               "steps": [{"range": [0, config.SCORING["medium_threshold"]], "color": "rgba(34,197,94,.22)"},
                         {"range": [config.SCORING["medium_threshold"], config.SCORING["high_threshold"]],
                          "color": "rgba(245,158,11,.22)"},
                         {"range": [config.SCORING["high_threshold"], 100], "color": "rgba(239,68,68,.22)"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": UI_TEXT, "family": "Inter, Segoe UI, system-ui, sans-serif"})
    return fig


def render_result(result: dict) -> None:
    a = result["assessment"]
    css = risk_css(a.category)
    color = config.COLORS[a.category]

    st.markdown("#### Extracted / analyzed URL (shown as text only - never opened)")
    st.code(a.url, language=None)

    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(render_gauge(a.final_score, a.category), **stretch(st.plotly_chart))
    with right:
        st.markdown(f"<span class='badge {css}'>{a.category}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='bar'><span style='width:{a.final_score}%;background:{color}'></span></div>",
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("Final score", a.final_score, risk_text_css(a.category)), unsafe_allow_html=True)
        c2.markdown(card("Rule score", a.rule_score, "c-blue"), unsafe_allow_html=True)
        c3.markdown(card("ML prediction", a.ml_label or "N/A",
                         {"SAFE": "c-green", "SUSPICIOUS": "c-orange", "MALICIOUS": "c-red"}.get(a.ml_label, "c-cyan"),
                         f"ML score {a.ml_score}" if a.ml_score is not None else "model unavailable"),
                    unsafe_allow_html=True)

    st.markdown(f"<div class='warn'>⚠️ {html.escape(a.disclaimer)}</div>", unsafe_allow_html=True)

    if result.get("model_error"):
        st.warning(f"ML model unavailable ({result['model_error']}). The score uses rules only.")
    if result.get("message") and "assumed" in result["message"]:
        st.info(result["message"])

    tabs = st.tabs(["🔎 Indicators", "🤖 ML prediction", "📚 RAG explanation",
                    "🧮 Score logic", "✅ Recommended actions"])

    with tabs[0]:
        if a.indicators:
            for ind in sorted(a.indicators, key=lambda i: -i["points"]):
                st.markdown(
                    f"<div class='ind {html.escape(ind['severity'])}'><b>{html.escape(ind['name'])}</b> "
                    f"(+{ind['points']} pts)<br><span>{html.escape(ind['detail'])}</span></div>",
                    unsafe_allow_html=True)
        else:
            st.success("No suspicious indicators were triggered by the rule engine.")
        with st.expander(f"Checks that passed ({len(a.passed_checks)})"):
            st.write(", ".join(a.passed_checks) or "None")

    with tabs[1]:
        if a.ml_available:
            probs = pd.DataFrame({"Class": list(a.ml_probabilities.keys()),
                                  "Probability": list(a.ml_probabilities.values())})
            fig = px.bar(probs, x="Class", y="Probability", color="Class", range_y=[0, 1],
                         color_discrete_map={k: config.COLORS[k] for k in config.ML_LABELS})
            fig.update_layout(height=300, showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font={"color": UI_TEXT, "family": "Inter, Segoe UI, system-ui, sans-serif"})
            st.plotly_chart(fig, **stretch(st.plotly_chart))
            st.caption("The demo model is trained on a SYNTHETIC dataset, so these probabilities "
                       "are illustrative and not real-world detection confidence.")
        else:
            st.warning("The ML model is not available.")

    with tabs[2]:
        exp = result.get("explanation")
        if not exp:
            st.warning(f"RAG assistant unavailable: {result.get('rag_error') or 'unknown error'}")
        else:
            st.markdown("<div class='note'>📌 <b>Evidence about THIS URL</b> comes from the rule engine "
                        "and ML model. <b>Background knowledge</b> below is retrieved from the "
                        "knowledge base and is general information, not proof about this link.</div>",
                        unsafe_allow_html=True)
            st.markdown("##### 🔬 Evidence found in this URL")
            for line in exp["url_evidence"] or ["No suspicious indicators were found."]:
                st.markdown(f"- {line}")
            st.markdown("##### 📖 Background knowledge (retrieved from the knowledge base)")
            for k in exp["retrieved_knowledge"]:
                st.markdown(f"**{k['indicator']}** - {k['passage']}  \n"
                            f"<span class='src'>source: {html.escape(k['source'])} · similarity {k['score']}</span>",
                            unsafe_allow_html=True)
            if not exp["retrieved_knowledge"]:
                st.write("No indicator-specific passages were needed.")
            st.markdown("##### 🛡️ General safety advice (retrieved)")
            for k in exp["safety_advice"]:
                st.markdown(f"- {k['passage']}  <span class='src'>({html.escape(k['source'])})</span>",
                            unsafe_allow_html=True)
            info = exp["info"]
            st.caption(f"Embeddings: {info['backend']} · {info['generation_mode']}.")

    with tabs[3]:
        st.markdown("The score is computed transparently from `config.py`:")
        for step in a.score_breakdown:
            st.markdown(f"- {step}")
        st.json(config.SCORING, expanded=False)

    with tabs[4]:
        for rec in a.recommendations:
            st.markdown(f"- {rec}")

    report = report_generator.build_report(a, result.get("explanation"))
    st.download_button("⬇️ Download security report (Markdown)", report,
                       file_name="security_report.md", mime="text/markdown")


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_scanner() -> None:
    st.header("📷 QR Code Scanner")
    st.markdown("<div class='note'>Upload a QR code image. The code is decoded locally; the link "
                "inside is <b>never opened or downloaded</b>.</div>", unsafe_allow_html=True)
    upload = st.file_uploader("Upload QR code image", type=config.ALLOWED_IMAGE_EXTENSIONS)
    if upload is None:
        st.info("No image uploaded yet. Sample images can be created with `python generate_sample_qr.py`.")
        return

    data = upload.getvalue()
    try:
        image = validate_image_bytes(data, upload.name)       # Pillow validation
    except ImageValidationError as exc:
        st.error(f"❌ Invalid image: {exc}")
        return

    col_img, col_res = st.columns([1, 1.4])
    with col_img:
        st.image(image, caption=f"{upload.name} · {image.width}x{image.height}px",
                 **stretch(st.image))
    scan = scan_qr_image(image)                               # OpenCV (+ Pyzbar)

    with col_res:
        if not scan.success:
            st.error(f"❌ QR decoding failed. {scan.error}")
            return
        st.success(f"Decoded {len(scan.contents)} QR code(s) using: {', '.join(scan.decoders_used)}")
        if not PYZBAR_AVAILABLE:
            st.caption("Pyzbar/ZBar is not available on this machine; OpenCV is used instead.")
        valid = []
        for i, content in enumerate(scan.contents, 1):
            ok, normalized, message = validate_url(content)
            st.markdown(f"**QR #{i}** - {'✅ valid URL' if ok else '⚠️ not a usable URL'}")
            st.code(content, language=None)
            if ok:
                valid.append(content)
            else:
                st.warning(message)
        if not valid:
            return
        choice = valid[0] if len(valid) == 1 else st.selectbox("Choose which URL to analyze", valid)
        if st.button("🔍 Analyze URL", type="primary"):
            with st.spinner("Analyzing..."):
                result, _ = run_analysis(choice, source="qr_upload")
            if result is None:
                st.error("Analysis failed.")

    latest = st.session_state.get("latest")
    if latest:
        st.divider()
        render_result(latest)


def page_url_analysis() -> None:
    st.header("🔗 URL Analysis")
    st.markdown("<div class='note'>Paste any link to analyze its <b>text</b>. The link is not visited.</div>",
                unsafe_allow_html=True)
    examples = {"(choose an example)": "",
                "Normal HTTPS": "https://www.example.com/about",
                "HTTP": "http://www.example.com/welcome",
                "IP-based": "http://203.0.113.5:8080/login/verify.php",
                "Suspicious look-alike": "http://paypa1-secure-login.example.test/verify?redirect=http://x.example.test",
                "@ trick": "https://www.bank.example.com@secure-update.example.test/account"}
    pick = st.selectbox("Harmless test examples (reserved domains)", list(examples))
    url = st.text_input("URL to analyze", value=examples[pick], max_chars=config.MAX_URL_LENGTH_ACCEPTED)
    if st.button("🔍 Analyze", type="primary"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            result, message = run_analysis(url, source="manual")
            if result is None:
                st.error(f"❌ {message}")
    latest = st.session_state.get("latest")
    if latest:
        st.divider()
        render_result(latest)


def page_rag() -> None:
    st.header("📚 RAG Cybersecurity Assistant")
    engine, err = get_rag()
    if engine is None:
        st.error(f"The RAG engine could not start: {err}")
        return
    info = engine.info()
    st.markdown("<div class='note'>This assistant uses <b>Retrieval-Augmented Generation</b>: your question is "
                "embedded, <b>FAISS</b> finds the closest knowledge-base passages, and the answer is assembled "
                "from them. It is <b>retrieval-based generation - no LLM is running</b>.</div>",
                unsafe_allow_html=True)
    st.caption(f"Index: {info['chunks']} chunks from {info['documents']} documents · embeddings: "
               f"{info['backend']} ({info['model']})")
    if info["notice"]:
        st.warning(info["notice"])

    quick = ["What is phishing?", "How does QR code phishing work?", "How can I identify a malicious website?",
             "Why is the @ symbol in a URL dangerous?", "Does HTTPS mean a website is safe?",
             "How do I stay safe online?"]
    cols = st.columns(3)
    for i, q in enumerate(quick):
        if cols[i % 3].button(q, key=f"quick{i}", **stretch(st.button)):
            st.session_state["rag_q"] = q
    question = st.text_input("Ask a cybersecurity question", value=st.session_state.get("rag_q", ""),
                             max_chars=300)
    if question.strip():
        res = engine.answer_question(question.strip())
        st.markdown(res["answer"])
        if res["sources"]:
            st.markdown("##### Retrieved passages")
            for i, s in enumerate(res["sources"], 1):
                with st.expander(f"[{i}] {s['source']} · similarity {s['score']}"):
                    st.write(s["text"])
        st.caption("Retrieved text is general knowledge, not evidence about any specific URL.")

    latest = st.session_state.get("latest")
    if latest and latest.get("explanation"):
        st.divider()
        st.markdown("##### Explain the latest scan")
        exp = latest["explanation"]
        st.write("**Evidence about the scanned URL:**")
        for line in exp["url_evidence"] or ["No suspicious indicators."]:
            st.markdown(f"- {line}")
        st.write("**Background knowledge:**")
        for k in exp["retrieved_knowledge"]:
            st.markdown(f"- *{k['indicator']}*: {k['passage']}")


def page_dashboard() -> None:
    st.header("📊 Security Dashboard")
    stats = history_manager.get_statistics()
    c = st.columns(4)
    c[0].markdown(card("Total URLs analyzed", stats["total"], "c-cyan"), unsafe_allow_html=True)
    c[1].markdown(card("Safe (low risk)", stats["safe"], "c-green"), unsafe_allow_html=True)
    c[2].markdown(card("Suspicious (medium)", stats["suspicious"], "c-orange"), unsafe_allow_html=True)
    c[3].markdown(card("Malicious (high)", stats["malicious"], "c-red"), unsafe_allow_html=True)
    st.write("")

    df = history_manager.load_history()
    left, right = st.columns(2)
    with left:
        st.subheader("Risk distribution")
        if len(df):
            counts = df["risk_category"].value_counts().reset_index()
            counts.columns = ["Category", "Count"]
            fig = px.pie(counts, names="Category", values="Count", hole=0.55, color="Category",
                         color_discrete_map={k: config.COLORS[k] for k in
                                             ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]})
            fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": UI_TEXT, "family": "Inter, Segoe UI, system-ui, sans-serif"},
                              legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
            st.plotly_chart(fig, **stretch(st.plotly_chart))
        else:
            st.info("No scans yet. Analyze a URL to see charts.")
    with right:
        st.subheader("Score histogram")
        if len(df):
            fig2, ax = plt.subplots(figsize=(5, 3.2))
            fig2.patch.set_facecolor(UI_BG)
            ax.set_facecolor(UI_PANEL)
            ax.hist(pd.to_numeric(df["final_score"], errors="coerce").dropna(), bins=10,
                    range=(0, 100), color=UI_TEAL, edgecolor=UI_BG)
            ax.set_xlabel("Final risk score", color=UI_TEXT)
            ax.set_ylabel("Scans", color=UI_TEXT)
            ax.tick_params(colors=UI_MUTED)
            for spine in ax.spines.values():
                spine.set_color(UI_LINE)
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Waiting for data.")

    st.subheader("Latest scan result")
    if stats["latest"]:
        lt = stats["latest"]
        st.markdown(f"<span class='badge {risk_css(lt['risk_category'])}'>{html.escape(str(lt['risk_category']))}"
                    f"</span> &nbsp; score **{lt['final_score']}** · {html.escape(str(lt['timestamp']))}",
                    unsafe_allow_html=True)
        st.code(str(lt["url"]), language=None)
    else:
        st.info("Nothing scanned yet.")

    st.divider()
    st.subheader("🤖 Machine-learning model evaluation")
    metrics = ml_classifier.load_metrics()
    if metrics:
        st.markdown("<div class='warn'>These metrics were measured on the bundled <b>synthetic demo dataset</b>. "
                    "They show the training pipeline works and are <b>not</b> real-world detection accuracy.</div>",
                    unsafe_allow_html=True)
        m = st.columns(4)
        m[0].metric("Accuracy", metrics["accuracy"])
        m[1].metric("Precision (weighted)", metrics["precision_weighted"])
        m[2].metric("Recall (weighted)", metrics["recall_weighted"])
        m[3].metric("F1-score (weighted)", metrics["f1_weighted"])
        a, b = st.columns(2)
        with a:
            cm = px.imshow(metrics["confusion_matrix"], x=metrics["labels"], y=metrics["labels"],
                           text_auto=True, color_continuous_scale=[[0, UI_PANEL2], [1, UI_TEAL]],
                           labels=dict(x="Predicted", y="Actual", color="Count"))
            cm.update_layout(title="Confusion matrix (test split)", height=340,
                             paper_bgcolor="rgba(0,0,0,0)", font={"color": UI_TEXT, "family": "Inter, Segoe UI, system-ui, sans-serif"})
            st.plotly_chart(cm, **stretch(st.plotly_chart))
        with b:
            fi = pd.DataFrame(list(metrics["feature_importance"].items())[:10], columns=["Feature", "Importance"])
            fig3 = px.bar(fi.iloc[::-1], x="Importance", y="Feature", orientation="h",
                          title="Top feature importances",
                          color_discrete_sequence=[UI_PURPLE])
            fig3.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": UI_TEXT, "family": "Inter, Segoe UI, system-ui, sans-serif"})
            st.plotly_chart(fig3, **stretch(st.plotly_chart))
    else:
        st.info("No metrics yet. Train the model below.")

    st.subheader("⚙️ System maintenance")
    b1, b2 = st.columns(2)
    if b1.button("Retrain ML model (demo dataset)"):
        with st.spinner("Training..."):
            try:
                if not config.DATASET_PATH.exists():
                    ml_classifier.generate_sample_dataset()
                ml_classifier.train_model()
                get_model.clear()
                st.success("Model retrained.")
                st.rerun()
            except Exception as exc:
                st.error(f"Training failed: {exc}")
    if b2.button("Rebuild FAISS knowledge index"):
        with st.spinner("Building index..."):
            try:
                engine = RAGEngine()
                engine.build_index()
                get_rag.clear()
                st.success("Index rebuilt.")
                st.rerun()
            except Exception as exc:
                st.error(f"Index build failed: {exc}")


def page_history() -> None:
    st.header("🕑 Detection History")
    df = history_manager.load_history()
    if df.empty:
        st.info("No scans stored yet.")
        return
    st.caption("Stored locally in data/history.csv. URLs are sanitised (credentials and query values removed).")
    st.dataframe(df, hide_index=True, **stretch(st.dataframe))
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns([1, 1])
    c1.download_button("⬇️ Download history CSV", csv_bytes, "history.csv", "text/csv")
    confirm = c2.checkbox("I want to permanently delete my local history")
    if c2.button("🗑️ Clear history", disabled=not confirm):
        history_manager.clear_history()
        st.session_state.pop("latest", None)
        st.success("History cleared.")
        st.rerun()


def page_about() -> None:
    st.header("ℹ️ About Project")
    st.markdown(f"""
**{config.APP_TITLE}** - {config.APP_SUBTITLE}

**Problem.** Attackers hide malicious links in QR codes ("quishing"). People cannot read a QR code,
so they scan first and find out later. This system extracts the link, analyses it, and explains the
risk *before* anyone visits it.

**Pipeline.** QR image → Pillow validation → OpenCV/Pyzbar decoding → URL validation →
rule-based engine + Random-Forest classifier → transparent final score → RAG explanation
(Sentence-Transformers/TF-IDF embeddings + FAISS retrieval) → history & dashboard.

**Design principles.**
- The URL is never opened, fetched or downloaded; QR content is treated as untrusted input.
- The RAG component only *explains*. It cannot change the score or declare a URL safe.
- The ML model is trained on a synthetic demo dataset - replace `dataset/urls.csv` with real data for real use.
- A LOW RISK result never guarantees safety.

**Technologies.** Python, Streamlit, Pillow, OpenCV, Pyzbar, Sentence-Transformers, FAISS,
scikit-learn, Pandas, NumPy, Matplotlib, Plotly, regular expressions and URL parsing.
""")
    engine, _ = get_rag()
    bundle, _ = get_model()
    st.subheader("System status")
    s = st.columns(4)
    s[0].markdown(card("ML model", "Ready" if bundle else "Missing", "c-green" if bundle else "c-red"), unsafe_allow_html=True)
    s[1].markdown(card("RAG index", "Ready" if engine else "Missing", "c-green" if engine else "c-red"), unsafe_allow_html=True)
    s[2].markdown(card("Embeddings", engine.backend if engine else "-", "c-cyan"), unsafe_allow_html=True)
    s[3].markdown(card("Pyzbar", "Available" if PYZBAR_AVAILABLE else "OpenCV only",
                       "c-green" if PYZBAR_AVAILABLE else "c-orange"), unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
PAGES = {
    "📷 QR Code Scanner": page_scanner,
    "🔗 URL Analysis": page_url_analysis,
    "📚 RAG Cybersecurity Assistant": page_rag,
    "📊 Security Dashboard": page_dashboard,
    "🕑 Detection History": page_history,
    "ℹ️ About Project": page_about,
}


def main() -> None:
    load_css()

    # ---- Top navigation bar (replaces the old left sidebar; same options, same order) ----
    brand_col, nav_col = st.columns([0.8, 4.7])
    with brand_col:
        st.markdown("<div class='brand'><span class='brand-logo'>🛡️</span><div>"
                    "<div class='brand-name'>Link Analyzer</div>"
                    "<div class='brand-sub'>QR phishing &amp; malicious URL detection</div></div></div>",
                    unsafe_allow_html=True)
    with nav_col:
        page = st.radio("Navigation", list(PAGES), horizontal=True, label_visibility="collapsed")

    # ---- Header section (project name + description unchanged) ----
    st.markdown(f"<div class='hero'><div class='hero-text'><h1><span class='emo'>🛡️</span> {html.escape(config.APP_TITLE)}</h1>"
                f"<p>{html.escape(config.APP_SUBTITLE)}</p></div>"
                "<div class='chips'><span class='chip'>🔒 Links are analyzed as text only and are never opened.</span>"
                "</div></div>", unsafe_allow_html=True)
    PAGES[page]()

main()
