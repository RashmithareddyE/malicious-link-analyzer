from modules import history_manager, ml_classifier
from modules.rag_engine import RAGEngine, chunk_document
from modules.risk_assessment import assess
from modules.url_analyzer import analyze_url


def test_train_and_predict(tmp_path):
    ds = ml_classifier.generate_sample_dataset(tmp_path / "u.csv", rows_per_class=120)
    metrics = ml_classifier.train_model(ds, tmp_path / "m.joblib", tmp_path / "m.json")
    assert 0 <= metrics["accuracy"] <= 1 and len(metrics["confusion_matrix"]) == 3
    bundle = ml_classifier.load_model(tmp_path / "m.joblib")
    out = ml_classifier.predict_url("http://203.0.113.5:8080/login.php", bundle)
    assert out["label"] in ("SAFE", "SUSPICIOUS", "MALICIOUS")


def test_missing_model_and_dataset(tmp_path):
    import pytest
    with pytest.raises(ml_classifier.ModelNotAvailableError):
        ml_classifier.load_model(tmp_path / "nope.joblib")
    with pytest.raises(FileNotFoundError):
        ml_classifier.load_dataset(tmp_path / "nope.csv")


def test_chunking_keeps_heading():
    chunks = chunk_document({"source": "x.txt", "text": "Heading here\n\nFirst paragraph about phishing."})
    assert chunks and chunks[0]["text"].startswith("Heading here")


def test_rag_retrieves_relevant_source():
    engine = RAGEngine()
    engine.build_index(save=False)
    hits = engine.search("what is QR code phishing quishing", k=3)
    assert hits and hits[0]["source"] == "qr_phishing.txt"
    exp = engine.explain_url([{"rule_id": "at_symbol", "name": "'@' symbol", "points": 25, "detail": "d"}])
    assert exp["url_evidence"] and exp["retrieved_knowledge"]


def test_history_sanitises_and_clears(tmp_path):
    path = tmp_path / "h.csv"
    a = assess(analyze_url("http://user:secret@example.test/a?token=abc123"), None)
    rec = history_manager.add_record(a, path=path)
    assert "secret" not in rec["url"] and "abc123" not in rec["url"]
    assert history_manager.get_statistics(path)["total"] == 1
    history_manager.clear_history(path)
    assert history_manager.get_statistics(path)["total"] == 0
