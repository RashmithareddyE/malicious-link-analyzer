"""
config.py - Central configuration for the Malicious Link Analyzer.

Every tunable value (paths, scoring weights, thresholds, keyword lists) lives
here so the scoring logic is transparent and easy to change.
All paths are built with pathlib so they work on Windows, macOS and Linux.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root, where app.py lives)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "urls.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "url_classifier.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FAISS_INDEX_PATH = MODELS_DIR / "faiss.index"
CHUNKS_PATH = MODELS_DIR / "rag_chunks.json"
RAG_META_PATH = MODELS_DIR / "rag_meta.json"
TFIDF_PATH = MODELS_DIR / "rag_tfidf.joblib"
ASSETS_DIR = BASE_DIR / "assets"
STYLE_PATH = ASSETS_DIR / "style.css"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"
HISTORY_PATH = DATA_DIR / "history.csv"
SAMPLE_QR_DIR = BASE_DIR / "sample_qr_codes"

# --------------------------------------------------------------------------
# Upload / image validation
# --------------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "BMP"}   # real format checked by Pillow
MAX_UPLOAD_MB = 5
MAX_IMAGE_PIXELS = 25_000_000                    # protects against decompression bombs
MAX_URL_LENGTH_ACCEPTED = 4096                   # absolute cap for any decoded text

# --------------------------------------------------------------------------
# Rule-based detection thresholds
# --------------------------------------------------------------------------
LONG_URL_THRESHOLD = 75
VERY_LONG_URL_THRESHOLD = 150
MAX_NORMAL_DOTS = 4
MAX_NORMAL_HYPHENS_IN_DOMAIN = 2
MAX_NORMAL_SUBDOMAINS = 2
TYPOSQUAT_SIMILARITY = 0.80                      # difflib ratio (0-1)

STANDARD_PORTS = {80, 443}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc", "rb.gy",
    "lnkd.in", "bl.ink", "s.id", "v.gd", "qr.ae", "t.ly",
}

SUSPICIOUS_TLDS = {
    "zip", "mov", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link",
    "work", "support", "country", "kim", "loan", "men", "icu", "monster",
}

SUSPICIOUS_FILE_EXTENSIONS = {
    "exe", "scr", "bat", "cmd", "msi", "apk", "jar", "vbs", "js", "ps1",
    "zip", "rar", "7z", "iso", "dmg", "docm", "xlsm", "pif", "lnk", "hta",
}

PHISHING_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "account", "update",
    "secure", "security", "banking", "bank", "confirm", "password", "passwd",
    "wallet", "billing", "invoice", "payment", "suspend", "suspended",
    "unlock", "recover", "restore", "alert", "urgent", "support", "helpdesk",
    "credential", "authenticate", "reward", "prize", "gift", "free", "claim",
    "refund", "otp", "kyc",
]

REDIRECT_INDICATORS = [
    "redirect=", "redirect_uri=", "redirect_url=", "redir=", "url=", "next=",
    "goto=", "return=", "returnurl=", "continue=", "dest=", "destination=",
    "target=", "forward=", "link=",
]

SPECIAL_CHARACTERS = set("@!$%^*()+=[]{}|\\<>`~;,'\"")

# Popular brands / domains used for look-alike (typosquatting) detection.
# Only the registrable "name" part (without TLD) is compared.
POPULAR_BRANDS = [
    "google", "facebook", "instagram", "whatsapp", "amazon", "apple", "microsoft",
    "paypal", "netflix", "linkedin", "twitter", "youtube", "github", "dropbox",
    "outlook", "office365", "yahoo", "ebay", "walmart", "flipkart", "paytm",
    "phonepe", "gpay", "hdfcbank", "icicibank", "sbi", "axisbank", "chase",
    "wellsfargo", "bankofamerica", "citibank", "coinbase", "binance", "steam",
    "spotify", "telegram", "snapchat", "adobe", "zoom", "dhl", "fedex", "irs",
]

# --------------------------------------------------------------------------
# Rule-engine point values (each triggered rule adds these points, capped at 100)
# --------------------------------------------------------------------------
RULE_POINTS = {
    "ip_address": 30,
    "no_https": 15,
    "very_long_url": 15,
    "long_url": 8,
    "special_characters": 8,
    "url_shortener": 20,
    "suspicious_subdomains": 12,
    "typosquatting": 30,
    "phishing_keywords": 14,          # for 1-2 keywords; doubled for 3+
    "suspicious_extension": 20,
    "many_dots": 8,
    "at_symbol": 25,
    "redirect_indicator": 12,
    "punycode": 25,
    "unusual_port": 15,
    "multiple_hyphens": 10,
    "suspicious_tld": 12,
    "credentials_in_url": 20,
}

# --------------------------------------------------------------------------
# Final risk scoring (transparent + configurable)
# --------------------------------------------------------------------------
SCORING = {
    # Final score = rule_weight * rule_score + ml_weight * ml_score
    # (weights are re-normalised if the ML model is unavailable)
    "rule_weight": 0.60,
    "ml_weight": 0.40,
    # The ML score is the expected "danger" of the class probabilities:
    #   P(SAFE)*0 + P(SUSPICIOUS)*ml_suspicious_value + P(MALICIOUS)*100
    "ml_suspicious_value": 50,
    # Category thresholds on the final 0-100 score
    "medium_threshold": 30,
    "high_threshold": 60,
    # Safety floors: certain evidence can never produce a LOW RISK result
    "floor_if_ml_malicious": 35,       # ML says MALICIOUS -> at least MEDIUM
    "floor_if_critical_rule": 35,      # a critical rule fired -> at least MEDIUM
    "floor_if_rule_high": 60,          # rule score >= 70 -> at least HIGH
    "rule_high_cutoff": 70,
}

# Rules considered "critical" (a single one is enough to avoid LOW RISK)
CRITICAL_RULES = {"ip_address", "at_symbol", "typosquatting", "punycode",
                  "suspicious_extension", "credentials_in_url", "url_shortener"}

# --------------------------------------------------------------------------
# Machine learning
# --------------------------------------------------------------------------
ML_LABELS = ["SAFE", "SUSPICIOUS", "MALICIOUS"]
ML_RANDOM_STATE = 42
ML_TEST_SIZE = 0.2
ML_N_ESTIMATORS = 200
DATASET_ROWS_PER_CLASS = 600

# --------------------------------------------------------------------------
# RAG
# --------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RAG_CHUNK_SIZE = 600          # characters per chunk (approx.)
RAG_CHUNK_OVERLAP = 100
RAG_TOP_K = 4
RAG_MIN_SCORE = 0.05          # ignore retrieved chunks below this cosine score
# If True and the Sentence-Transformers model cannot be loaded (e.g. offline
# first run), fall back to a local TF-IDF embedder so the app still works.
RAG_ALLOW_TFIDF_FALLBACK = True

# --------------------------------------------------------------------------
# UI colours
# --------------------------------------------------------------------------
COLORS = {
    "LOW RISK": "#22c55e",
    "MEDIUM RISK": "#f59e0b",
    "HIGH RISK": "#ef4444",
    "SAFE": "#22c55e",
    "SUSPICIOUS": "#f59e0b",
    "MALICIOUS": "#ef4444",
    "accent": "#22d3ee",
    "primary": "#3b82f6",
}

APP_TITLE = "Malicious Link Analyzer"
APP_SUBTITLE = "An Intelligent QR Code-Based Phishing and Malicious URL Detection System Using RAG"
