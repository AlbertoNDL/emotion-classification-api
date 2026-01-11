def test_metrics_endpoint(client):
    r = client.get("/v1/internal/metrics")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict)
