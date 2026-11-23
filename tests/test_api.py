"""API smoke tests (need fastapi + httpx installed: `pip install -r requirements.txt`)."""
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_cgm

client = TestClient(app)


def _payload(days=14, **opts):
    df = make_cgm(days=days)
    return {
        "user_id": "u_test_01",
        "readings": [{"timestamp": t.isoformat(), "value_mg_dl": float(g)}
                     for t, g in zip(df["timestamp"], df["glucose"])],
        "options": opts,
    }


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_summary_ok():
    r = client.post("/api/v1/analytics/summary", json=_payload(timezone="Africa/Cairo"))
    assert r.status_code == 200
    body = r.json()
    assert body["meets_consensus_criteria"] is True
    assert abs(sum(body["time_in_range"][k] for k in
        ["very_low_pct", "low_pct", "in_range_pct", "high_pct", "very_high_pct"]) - 100) < 0.1


def test_all_values_missing_returns_422():
    p = {"user_id": "u_test_01", "readings": [{"timestamp": "2026-01-01T00:00:00Z", "value_mg_dl": None}]}
    r = client.post("/api/v1/analytics/summary", json=p)
    assert r.status_code == 422 and r.json()["error"] == "insufficient_data"


def test_bad_timezone_and_user_id_rejected():
    assert client.post("/api/v1/analytics/summary", json=_payload(timezone="Mars/Base")).status_code == 422
    p = _payload(); p["user_id"] = "john.doe@mail.com"
    assert client.post("/api/v1/analytics/summary", json=p).status_code == 422


def test_features_and_agp():
    assert client.post("/api/v1/preprocess/features", json=_payload()).status_code == 200
    assert len(client.post("/api/v1/analytics/agp", json=_payload()).json()["profile"]) == 24


def test_patterns_detect_and_flags():
    r = client.post("/api/v1/patterns/detect", json=_payload())
    assert r.status_code == 200
    b = r.json()
    assert b["disclaimer"].startswith("For informational purposes only")
    assert b["clustering"]["status"] in {"ok", "no_distinct_structure"}
    r2 = client.post("/api/v1/patterns/detect",
                     json={**_payload(), "pattern_options": {"include_clusters": False}})
    assert r2.json()["clustering"] is None and r2.json()["patterns"] is not None
    bad = client.post("/api/v1/patterns/detect",
                      json={**_payload(), "pattern_options": {"max_clusters": 99}})
    assert bad.status_code == 422


def test_anomalies_detect_and_options():
    p = _payload()
    p["readings"][1000]["value_mg_dl"] = 45.0
    for i in range(1001, 1007):
        p["readings"][i]["value_mg_dl"] = 45.0           # ~35 min below 54 mg/dL
    r = client.post("/api/v1/anomalies/detect", json=p)
    assert r.status_code == 200
    b = r.json()
    assert b["events"][0]["severity"] == "priority" and b["events"][0]["primary_type"] == "urgent_low"
    assert "not a real-time alarm" in b["note"] and b["disclaimer"].startswith("For informational")
    r2 = client.post("/api/v1/anomalies/detect",
                     json={**p, "anomaly_options": {"include_statistical": False}})
    assert r2.json()["method"]["mode"] == "rules_only"
    assert client.post("/api/v1/anomalies/detect",
                       json={**p, "anomaly_options": {"contamination": 0.9}}).status_code == 422
