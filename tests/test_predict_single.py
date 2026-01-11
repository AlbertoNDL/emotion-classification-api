def test_predict_single_ok(client, auth_headers):
    r = client.post(
        "/v1/predict-emotion",
        headers=auth_headers,
        json={"text": "I am very happy today"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "emotion" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_single_empty_text(client, auth_headers):
    r = client.post(
        "/v1/predict-emotion",
        headers=auth_headers,
        json={"text": "   "},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["emotion"] == "neutral"
    assert body["confidence"] == 1.0


def test_predict_single_no_api_key(client):
    r = client.post(
        "/v1/predict-emotion",
        json={"text": "test"},
    )
    assert r.status_code in (401, 403)
