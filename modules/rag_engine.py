"""
rag_engine.py - Retrieval-Augmented Generation over a local cybersecurity knowledge base.

Pipeline (all local, no API key needed)
---------------------------------------
1. LOAD     every .txt file in knowledge_base/
2. CHUNK    documents into passages (paragraph-aware, with the nearest heading kept)
3. EMBED    chunks with Sentence-Transformers (all-MiniLM-L6-v2). If that model cannot
            be loaded (for example on a first run with no internet), a local TF-IDF
            embedder is used instead and the app says so clearly.
4. STORE    normalised vectors in a FAISS index (inner product == cosine similarity)
5. RETRIEVE the passages closest to a question or to the detected URL indicators
6. GENERATE a grounded answer by selecting the most relevant sentences from the
            retrieved passages and arranging them in a template.

HONESTY NOTE: step 6 is RETRIEVAL-BASED (extractive) generation. No large language
model runs in this project. Every sentence shown comes from a knowledge-base file.

The retrieved text is general cybersecurity KNOWLEDGE. It is never evidence about the
analyzed URL and it never decides whether a URL is safe - that comes only from the
rule engine and the ML model (see risk_assessment.py).

Command line (project root):
    python -m modules.rag_engine build
    python -m modules.rag_engine query "What is quishing?"
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

import config

try:
    import faiss  # type: ignore
except ImportError as _exc:  # pragma: no cover
    faiss = None
    _FAISS_ERROR = _exc

# Environment switch: set MLA_EMBEDDER=tfidf to skip Sentence-Transformers
# (useful for offline use and automated tests).
ENV_EMBEDDER = "MLA_EMBEDDER"

# Which knowledge topic to search for when a given rule fires
RULE_TOPICS: Dict[str, str] = {
    "ip_address": "malicious URL uses an IP address instead of a domain name",
    "no_https": "HTTP without S is not encrypted, never enter passwords on HTTP pages",
    "very_long_url": "very long URL can hide the real destination",
    "long_url": "very long URL can hide the real destination",
    "special_characters": "indicators that a URL may be malicious special characters",
    "url_shortener": "URL shorteners and redirects hide the final destination",
    "suspicious_subdomains": "subdomain tricks, the real domain is the part before the domain ending",
    "typosquatting": "typosquatting look-alike domains imitate brand names",
    "phishing_keywords": "words like login verify secure update bank are used by attackers to look official",
    "suspicious_extension": "links to files ending in exe apk scr zip can install malware",
    "many_dots": "many dots or several subdomain levels disguise the real domain",
    "at_symbol": "the @ symbol, browsers ignore everything before it in the address",
    "redirect_indicator": "redirect parameters send the visitor to a different site",
    "punycode": "homograph attack punycode xn-- internationalised domain names",
    "unusual_port": "unusual port numbers are rarely used by public websites",
    "multiple_hyphens": "multiple hyphens in a domain are typical of throw-away phishing domains",
    "suspicious_tld": "frequently abused low-cost domain endings",
    "credentials_in_url": "embedded username and password before the @ trick",
}

_STOPWORDS = set("""a an the of to in on for and or is are be it this that with as at by from your you
what how why do does can i we they not no any when which who whom into about than then so such""".split())


class RAGError(Exception):
    """Raised when the knowledge base or index cannot be built or used."""


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9@\-]+", text.lower())
            if len(w) >= 3 and w not in _STOPWORDS}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def load_documents(kb_dir: Path = config.KNOWLEDGE_BASE_DIR) -> List[Dict[str, str]]:
    """Read every .txt file in the knowledge base folder."""
    kb_dir = Path(kb_dir)
    if not kb_dir.exists():
        raise RAGError(f"Knowledge base folder not found: {kb_dir}")
    docs = []
    for path in sorted(kb_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[WARN] Could not read {path.name}: {exc}", file=sys.stderr)
            continue
        if text:
            docs.append({"source": path.name, "text": text})
    if not docs:
        raise RAGError(f"No .txt documents found in {kb_dir}. Add knowledge files and rebuild.")
    return docs


def chunk_document(doc: Dict[str, str], chunk_size: int = config.RAG_CHUNK_SIZE,
                   overlap: int = config.RAG_CHUNK_OVERLAP) -> List[Dict]:
    """Split a document into meaningful passages.

    Paragraphs are kept together where possible; a short paragraph without a final
    full stop is treated as a heading and prepended to the paragraphs after it.
    Over-long paragraphs are split on sentence boundaries with a small overlap.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc["text"]) if p.strip()]
    chunks: List[Dict] = []
    heading = ""
    buffer = ""

    def flush():
        nonlocal buffer
        if buffer.strip():
            body = buffer.strip()
            chunks.append({"source": doc["source"], "heading": heading,
                           "text": f"{heading}. {body}" if heading else body})
        buffer = ""

    for para in paragraphs:
        is_heading = len(para) < 80 and not para.endswith((".", "!", "?")) and "\n" not in para
        if is_heading:
            flush()
            heading = para
            continue
        if len(para) > chunk_size:
            flush()
            sentences = split_sentences(para)
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) + 1 > chunk_size:
                    buffer = current
                    flush()
                    tail = current[-overlap:] if overlap else ""
                    # start the next chunk at a sentence-ish boundary
                    current = (tail[tail.find(" ") + 1:] + " " if tail else "") + sent
                else:
                    current = f"{current} {sent}".strip()
            buffer = current
            flush()
        elif len(buffer) + len(para) + 1 > chunk_size:
            flush()
            buffer = para
        else:
            buffer = f"{buffer} {para}".strip()
    flush()
    return chunks


def build_chunks(kb_dir: Path = config.KNOWLEDGE_BASE_DIR) -> List[Dict]:
    """Load all documents and chunk them. Each chunk gets a numeric id."""
    all_chunks: List[Dict] = []
    for doc in load_documents(kb_dir):
        all_chunks.extend(chunk_document(doc))
    for i, ch in enumerate(all_chunks):
        ch["id"] = i
    if not all_chunks:
        raise RAGError("The knowledge base produced no text chunks.")
    return all_chunks


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------
class _SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit_encode(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.ascontiguousarray(emb, dtype=np.float32)


class _TfidfEmbedder:
    name = "tfidf-fallback"
    model_name = "TF-IDF (scikit-learn)"

    def __init__(self, vectorizer=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = vectorizer or TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)

    def fit_encode(self, texts: List[str]) -> np.ndarray:
        self.vectorizer.fit(texts)
        return self.encode(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(mat / norms, dtype=np.float32)


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class RAGEngine:
    """Builds, loads and queries the FAISS-backed knowledge index."""

    def __init__(self, kb_dir: Path = config.KNOWLEDGE_BASE_DIR):
        if faiss is None:
            raise RAGError("faiss-cpu is not installed. Run: pip install faiss-cpu "
                           f"(original error: {_FAISS_ERROR})")
        self.kb_dir = Path(kb_dir)
        self.chunks: List[Dict] = []
        self.index = None
        self.embedder = None
        self.backend: str = "not built"
        self.model_name: str = ""
        self.notice: str = ""

    # ---------------------------------------------------------------- build
    def _create_embedder(self):
        """Try Sentence-Transformers first; fall back to TF-IDF if allowed."""
        if os.environ.get(ENV_EMBEDDER, "").lower() == "tfidf":
            self.notice = f"Embedder forced to TF-IDF via the {ENV_EMBEDDER} environment variable."
            return _TfidfEmbedder()
        try:
            return _SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
        except Exception as exc:  # ImportError, OSError (no internet), etc.
            if not config.RAG_ALLOW_TFIDF_FALLBACK:
                raise RAGError(f"Could not load the Sentence-Transformers model: {exc}") from exc
            self.notice = ("Sentence-Transformers model unavailable "
                           f"({type(exc).__name__}); using the local TF-IDF fallback embedder. "
                           "Connect to the internet once and rebuild the index to use "
                           "Sentence-Transformers.")
            return _TfidfEmbedder()

    def build_index(self, save: bool = True) -> Dict:
        """Chunk the knowledge base, embed it and create the FAISS index."""
        self.chunks = build_chunks(self.kb_dir)
        self.embedder = self._create_embedder()
        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedder.fit_encode(texts)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])   # inner product on unit vectors
        self.index.add(embeddings)
        self.backend = self.embedder.name
        self.model_name = self.embedder.model_name
        if save:
            self._save()
        return self.info()

    def _save(self) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(config.FAISS_INDEX_PATH))
        config.CHUNKS_PATH.write_text(json.dumps(self.chunks, indent=1), encoding="utf-8")
        meta = {"backend": self.backend, "model_name": self.model_name,
                "n_chunks": len(self.chunks), "dim": int(self.index.d)}
        config.RAG_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if isinstance(self.embedder, _TfidfEmbedder):
            joblib.dump(self.embedder.vectorizer, config.TFIDF_PATH)
        elif config.TFIDF_PATH.exists():
            config.TFIDF_PATH.unlink()

    # ----------------------------------------------------------------- load
    def load(self) -> Dict:
        """Load a saved index, or build one if nothing usable is saved."""
        try:
            meta = json.loads(config.RAG_META_PATH.read_text(encoding="utf-8"))
            self.chunks = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
            self.index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            if self.index.ntotal != len(self.chunks):
                raise ValueError("index and chunks are out of sync")
            if meta["backend"] == "tfidf-fallback":
                self.embedder = _TfidfEmbedder(joblib.load(config.TFIDF_PATH))
            else:
                self.embedder = _SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME)
            self.backend, self.model_name = self.embedder.name, self.embedder.model_name
            return self.info()
        except Exception:
            # Missing / corrupted / model unavailable -> rebuild from the .txt files
            return self.build_index(save=True)

    def info(self) -> Dict:
        return {"backend": self.backend, "model": self.model_name,
                "chunks": len(self.chunks),
                "documents": len({c["source"] for c in self.chunks}),
                "notice": self.notice,
                "generation_mode": "Retrieval-based (extractive) generation - no LLM is used"}

    # ------------------------------------------------------------- retrieve
    def search(self, query: str, k: int = config.RAG_TOP_K) -> List[Dict]:
        """Return the top-k chunks (with cosine scores) for a query."""
        if self.index is None or self.embedder is None:
            raise RAGError("The index is not loaded. Call load() or build_index() first.")
        query = (query or "").strip()
        if not query:
            return []
        q = self.embedder.encode([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or float(score) < config.RAG_MIN_SCORE:
                continue
            chunk = dict(self.chunks[int(idx)])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)
        return results

    # ------------------------------------------------------------- generate
    @staticmethod
    def _best_sentences(text: str, query: str, n: int) -> List[str]:
        """Pick the n sentences with the most word overlap with the query (original order)."""
        sentences = split_sentences(text)
        q = _tokens(query)
        scored = []
        for i, s in enumerate(sentences):
            overlap = len(q & _tokens(s))
            scored.append((overlap, -i, i, s))
        top = sorted(scored, reverse=True)[:n]
        return [s for _, _, _, s in sorted(top, key=lambda t: t[2])]

    def answer_question(self, question: str, k: int = config.RAG_TOP_K) -> Dict:
        """Answer a general cybersecurity question from retrieved passages only."""
        hits = self.search(question, k)
        if not hits:
            return {
                "answer": ("I could not find relevant passages in the local knowledge base for "
                           "that question. Try asking about phishing, QR code phishing (quishing), "
                           "malicious URLs, URL or domain spoofing, HTTPS, or staying safe online."),
                "sources": [], "info": self.info()}
        # Keep only passages reasonably close to the best match (drops weak tail hits)
        hits = [h for h in hits if h["score"] >= 0.5 * hits[0]["score"]]
        lines = []
        for rank, hit in enumerate(hits[:3], start=1):
            picked = self._best_sentences(hit["text"], question, 2)
            lines.append(f"- {' '.join(picked)} *[{rank}]*")
        answer = ("**Answer assembled from the retrieved knowledge-base passages:**\n\n"
                  + "\n".join(lines))
        return {"answer": answer, "sources": hits, "info": self.info()}

    def explain_url(self, triggered: List[Dict], max_topics: int = 6) -> Dict:
        """Explain an analyzed URL.

        `triggered` is a list of dicts {rule_id, name, points, detail} from the rule engine.
        The result keeps two things strictly apart:
          - url_evidence: facts measured on THIS URL (from the rule engine, not from RAG)
          - retrieved_knowledge: general background retrieved from the knowledge base
        """
        top = sorted(triggered, key=lambda t: -t.get("points", 0))[:max_topics]
        url_evidence = [f"{t['name']}: {t['detail']}" for t in top]
        knowledge = []
        seen = set()
        for item in top:
            topic = RULE_TOPICS.get(item["rule_id"], item["name"])
            hits = self.search(topic, k=2)
            for hit in hits:
                if hit["id"] in seen:
                    continue
                seen.add(hit["id"])
                knowledge.append({
                    "indicator": item["name"],
                    "passage": " ".join(self._best_sentences(hit["text"], topic, 2)),
                    "source": hit["source"], "score": hit["score"]})
                break
        advice = []
        for hit in self.search("how to scan QR codes safely and stay safe online", k=2):
            advice.append({"passage": " ".join(self._best_sentences(
                hit["text"], "scan QR codes safely preview URL", 3)),
                "source": hit["source"], "score": hit["score"]})
        return {"url_evidence": url_evidence, "retrieved_knowledge": knowledge,
                "safety_advice": advice, "info": self.info()}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("build", "query"):
        print('Usage: python -m modules.rag_engine build | query "your question"')
        return 1
    try:
        engine = RAGEngine()
        if argv[0] == "build":
            info = engine.build_index()
            print(f"[OK] FAISS index built: {info['chunks']} chunks from {info['documents']} documents")
            print(f"     Embedding backend: {info['backend']} ({info['model']})")
            if info["notice"]:
                print(f"     NOTE: {info['notice']}")
            return 0
        engine.load()
        result = engine.answer_question(" ".join(argv[1:]))
        print(result["answer"])
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['source']}  (score {s['score']})")
        return 0
    except RAGError as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
