from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

ARTICLE_TEXT = (
    "Apple shares surged after the company posted record iPhone revenue and "
    "raised its full-year guidance, with analysts praising strong demand. "
    "Meanwhile, the Red Cross announced a new disaster relief partnership "
    "unrelated to the tech sector."
)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_returns_subjects_with_expected_shape():
    resp = client.post("/analyze", json={"text": ARTICLE_TEXT})
    assert resp.status_code == 200
    body = resp.json()
    assert body["financial_mode"] is False
    assert len(body["subjects"]) >= 1

    subject = body["subjects"][0]
    for key in ("subject", "label", "mentions", "summary", "sentiment", "financial_relevant"):
        assert key in subject
    assert set(subject["sentiment"].keys()) == {"compound", "label"}


def test_analyze_rejects_empty_text():
    resp = client.post("/analyze", json={"text": "   "})
    assert resp.status_code == 422


def test_analyze_financial_mode_via_body_field():
    resp = client.post("/analyze", json={"text": ARTICLE_TEXT, "financial_mode": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["financial_mode"] is True
    names = {s["subject"] for s in body["subjects"]}
    assert "the Red Cross" not in names


def test_analyze_financial_mode_via_query_param_overrides_body():
    resp = client.post(
        "/analyze?financial_mode=false",
        json={"text": ARTICLE_TEXT, "financial_mode": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["financial_mode"] is False
    names = {s["subject"] for s in body["subjects"]}
    assert "the Red Cross" in names


def test_analyze_url_rejects_unreachable_host():
    resp = client.post(
        "/analyze/url", json={"url": "http://this-domain-should-not-resolve.invalid/article"}
    )
    assert resp.status_code == 502
