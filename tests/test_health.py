def test_health_live_ok(client):
    r = client.get("/v1/health/live")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["inference"] == "ok"
