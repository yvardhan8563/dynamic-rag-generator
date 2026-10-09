import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings
from app.main import create_app


def test_collection_persistence_and_errors(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/health/live").status_code == 200
        assert client.get("/api/v1/health/ready").json()["storage"] == "ready"
        response = client.post("/api/v1/collections", json={"name": "  Test collection  "})
        assert response.status_code == 201
        collection = response.json()
        assert collection["name"] == "Test collection"
        assert collection["document_count"] == 0
        bad = client.post("/api/v1/collections", json={"name": " "})
        assert bad.status_code == 422
        assert bad.json()["error"]["request_id"] == bad.headers["X-Request-ID"]
        assert client.get("/api/v1/collections/00000000-0000-0000-0000-000000000000").status_code == 404
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/v1/collections").json()[0]["id"] == collection["id"]


def test_invalid_settings():
    with pytest.raises(ValidationError):
        Settings(chunk_tokens=30, chunk_overlap=30, _env_file=None)


def test_second_worker_rejected(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    with TestClient(create_app(settings)):
        with pytest.raises(RuntimeError, match="one worker"):
            with TestClient(create_app(settings)):
                pass
