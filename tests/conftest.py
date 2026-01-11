import pytest
from fastapi.testclient import TestClient
from app.main import app, get_batcher
from app.config import settings


@pytest.fixture(autouse=True)
def override_api_key():
    settings.API_KEY = "test-api-key"


class TestBatcher:
    def is_ready(self) -> bool:
        return True

    async def predict(self, text: str):
        # Single inference → List[float]
        return [0.1, 0.2, 0.7]

    async def predict_batch(self, texts):
        # Batch inference → List[List[float]]
        return [
            [0.1, 0.2, 0.7]
            for _ in texts
        ]


@pytest.fixture
def test_batcher():
    return TestBatcher()


@pytest.fixture
def client(test_batcher):
    app.dependency_overrides[get_batcher] = lambda: test_batcher

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-api-key"}
