# 🛡️ Malicious Link Analyzer
### An Intelligent QR Code-Based Phishing and Malicious URL Detection System Using RAG

## 1. Introduction
Malicious Link Analyzer reads a QR code image, extracts the link inside, analyses it with a rule engine and a
machine-learning classifier, retrieves relevant cybersecurity knowledge with **Retrieval-Augmented Generation (RAG)**,
and shows a clear, explainable risk assessment **before** anyone visits the website. The link is **never opened**.

## 2. Problem statement
Criminals hide phishing and malware links in QR codes ("quishing"). Users cannot read a QR code, so they scan first
and discover the danger later, exposing passwords, banking details and personal data.

## 3. Objectives
- Decode QR codes safely (Pillow + OpenCV + optional Pyzbar).
- Detect suspicious URL patterns with a transparent, configurable rule engine.
- Classify URLs as SAFE / SUSPICIOUS / MALICIOUS with a Random Forest.
- Explain *why* with a real RAG pipeline (Sentence-Transformers + FAISS).
- Combine everything into a 0-100 score and LOW / MEDIUM / HIGH category.
- Offer a professional Streamlit dashboard with local detection history.

## 4. Technologies
Python · Streamlit · Pillow · OpenCV · Pyzbar · Sentence-Transformers · FAISS · scikit-learn · Pandas · NumPy ·
Matplotlib · Plotly · Regular expressions & URL parsing. No paid API key is needed.

## 5. System architecture
```
 QR image ─► Pillow validation ─► OpenCV / Pyzbar decode ─► URL validation
                                                              │
                         ┌────────────────────────────────────┤
                         ▼                                    ▼
              Rule engine (17 checks)              Feature extractor ─► Random Forest
                   rule score                          ML label + ML score
                         └──────────────┬──────────────────────┘
                                        ▼
                       Risk assessment (weights, floors)  ──► final score + category
                                        │
              triggered indicators ─► RAG (embed ─► FAISS ─► passages ─► grounded text)
                                        │        (explains only; cannot change the score)
                                        ▼
                    Streamlit UI · history.csv · Markdown report
```

## 6. Folder structure
```
malicious_link_analyzer/
├── app.py                  Streamlit application
├── config.py               all settings, weights, thresholds, keyword lists
├── generate_sample_qr.py   creates harmless test QR images
├── requirements.txt
├── .streamlit/config.toml  dark theme
├── modules/
│   ├── qr_scanner.py       validation + decoding
│   ├── feature_extractor.py URL parsing + numeric features
│   ├── url_analyzer.py     URL validation + rule engine
│   ├── ml_classifier.py    dataset generation, training, prediction
│   ├── rag_engine.py       chunking, embeddings, FAISS, grounded answers
│   ├── risk_assessment.py  final scoring
│   ├── history_manager.py  CSV history (sanitised)
│   └── report_generator.py Markdown reports
├── knowledge_base/         5 cybersecurity .txt documents
├── dataset/urls.csv        SYNTHETIC demo dataset
├── models/                 generated model, metrics, FAISS index
├── assets/style.css
├── data/                   history.csv (created at runtime)
├── reports/
├── sample_qr_codes/
└── tests/
```

## 7. Installation (Windows, Command Prompt or VS Code terminal)
Run all commands from the project root folder (the one containing `app.py`).
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m modules.ml_classifier generate
python -m modules.ml_classifier train
python -m modules.rag_engine build
python generate_sample_qr.py
streamlit run app.py
```
(PowerShell: if activation is blocked run `Set-ExecutionPolicy -Scope Process Bypass` first.)
The first `rag_engine build` downloads the ~90 MB `all-MiniLM-L6-v2` model, so it needs internet once.
The app also trains the model and builds the index automatically if they are missing.

**Pyzbar on Windows:** if you see a ZBar/DLL warning, install the "Visual C++ Redistributable 2013" or ignore it –
the app falls back to OpenCV decoding automatically.

## 8. How RAG works here
1. **Load** the `.txt` files in `knowledge_base/`.
2. **Chunk** them by paragraph (≈600 characters, headings kept with their text).
3. **Embed** each chunk with Sentence-Transformers `all-MiniLM-L6-v2`.
4. **Store** the normalised vectors in a **FAISS** index (inner product = cosine similarity).
5. **Retrieve** the closest chunks for a user question or for each triggered URL indicator.
6. **Generate** an answer by selecting the most relevant sentences of the retrieved chunks.

This last step is **retrieval-based (extractive) generation – no LLM runs**, and every sentence comes from the
knowledge base. If the Sentence-Transformers model cannot be loaded (e.g. first run offline) the app uses a local
**TF-IDF fallback** embedder and states this in the UI. Retrieved text is *background knowledge*, shown separately
from *evidence about the scanned URL*. RAG never changes the score and can never declare a URL safe.

## 9. How the machine learning works
`dataset/urls.csv` → feature extraction (19 numeric features: length, dots, hyphens, digits, special characters,
HTTPS, IP, keywords, domain length, …) → stratified 80/20 train-test split → Random Forest → accuracy, precision,
recall, F1, confusion matrix (saved in `models/metrics.json`, shown on the dashboard).

> ⚠️ **The bundled dataset is SYNTHETIC** (generated from templates on reserved domains). Its near-perfect metrics only
> prove the pipeline works – they are **not real-world detection accuracy**. Use a real labelled dataset
> (`url,label` with SAFE/SUSPICIOUS/MALICIOUS) and retrain for realistic results.

## 10. Scoring
`final = 0.60 × rule score + 0.40 × ML score` (weights in `config.SCORING`), then safety floors (a critical rule such as
IP-address host, `@` trick, look-alike domain, punycode, executable download or URL shortener forces at least MEDIUM).
Categories: LOW < 30 ≤ MEDIUM < 60 ≤ HIGH. **A LOW RISK result does not guarantee a site is safe.**

## 11. Features / screenshots
Add your screenshots here: `![Scanner](screenshots/scanner.png)` `![Dashboard](screenshots/dashboard.png)`
- QR Code Scanner · URL Analysis · RAG Cybersecurity Assistant · Security Dashboard · Detection History · About

## 12. Testing
```bat
python generate_sample_qr.py
python -m pytest -q tests
```
Upload the images in `sample_qr_codes/` on the **QR Code Scanner** page:

| File | Expected result |
|---|---|
| 01_safe_https.png | LOW RISK, ML SAFE |
| 02_http_url.png | LOW/MEDIUM, "HTTP instead of HTTPS" flagged |
| 03_ip_based.png | HIGH RISK, IP address, unusual port, keywords, `.php` |
| 04_suspicious.png | HIGH/MEDIUM RISK, look-alike `paypa1`, redirect, keywords |
| 05_at_symbol.png | MEDIUM/HIGH, `@` trick flagged |
| 06_shortener.png | MEDIUM RISK, URL shortener |
| 07_multiple_qr.png | two URLs listed, choose which to analyze |
| 08_text_only.png | decoded but "not a usable URL" |
| 09_no_qr.png | error: no QR code detected |
| 10_invalid_damaged_qr.png | error: could not be decoded |
| 11_corrupt_file.png | error: not a valid image |

## 13. Limitations
- Only the URL text is analysed; no page content, DNS, WHOIS, certificate or reputation lookups.
- The demo ML model uses synthetic data; the rule list and brand list are small.
- Brand-name heuristics can give false positives (e.g. legitimate domains containing a brand name).
- Extractive RAG cannot reason beyond its knowledge base; it is not a chatbot.
- Heavily damaged or very small QR codes may fail to decode.

## 14. Future enhancements
Real phishing datasets (PhishTank/OpenPhish exports), safe sandboxed URL expansion, WHOIS/domain-age features,
live camera scanning, a local LLM (e.g. via Ollama) for generation on top of the same retrieval, multilingual
knowledge base, user feedback loop for retraining.
