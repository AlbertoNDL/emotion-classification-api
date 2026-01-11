def test_predict_batch_ok(client, auth_headers):
    r = client.post(
        "/v1/predict-emotion-batch",
        headers=auth_headers,
        json={"texts": ["I am happy", "I am sad"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) == 2


def test_predict_batch_empty(client, auth_headers):
    r = client.post(
        "/v1/predict-emotion-batch",
        headers=auth_headers,
        json={"texts": []},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
