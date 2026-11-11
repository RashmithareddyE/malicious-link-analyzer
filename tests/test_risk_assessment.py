import config
from modules.risk_assessment import assess, categorize
from modules.url_analyzer import analyze_url


def ml(label, score):
    return {"label": label, "ml_score": score,
            "probabilities": {"SAFE": 0.0, "SUSPICIOUS": 0.0, "MALICIOUS": 0.0}}


def test_low_risk_for_clean_url():
    a = assess(analyze_url("https://www.example.com"), ml("SAFE", 0))
    assert a.category == "LOW RISK" and a.final_score == 0
    assert "does not guarantee" in a.disclaimer


def test_high_risk_for_ip_login():
    a = assess(analyze_url("http://203.0.113.5:8080/login/verify.php"), ml("MALICIOUS", 90))
    assert a.category == "HIGH RISK"


def test_critical_rule_floor_prevents_low_risk():
    a = assess(analyze_url("https://google.com@evil.example.test/"), ml("SAFE", 0))
    assert a.category != "LOW RISK"


def test_works_without_ml_model():
    a = assess(analyze_url("http://203.0.113.5/x"), None)
    assert not a.ml_available and a.final_score >= config.SCORING["floor_if_critical_rule"]


def test_categorize_thresholds():
    assert categorize(10) == "LOW RISK"
    assert categorize(config.SCORING["medium_threshold"]) == "MEDIUM RISK"
    assert categorize(config.SCORING["high_threshold"]) == "HIGH RISK"


def test_rag_cannot_lower_score():
    # The assessment has no RAG input at all: the score depends only on rules + ML.
    a = assess(analyze_url("http://203.0.113.5/x"), ml("SUSPICIOUS", 50))
    assert "RAG explanations are informational only" in " ".join(a.score_breakdown)
