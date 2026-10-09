import pytest
from fastapi.testclient import TestClient

from app.errors import AppError
from app.main import create_app
from tests.conftest import FakeEmbedder, collection, upload


def test_isolation_duplicates_restart_and_deletion(settings):
    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        a, b = collection(client, "Policies"), collection(client, "Space")
        first = upload(client, a, "Refunds are available for thirty days.")
        assert first.status_code == 201
        assert upload(client, a, "Refunds are available for thirty days.").status_code == 409
        assert upload(client, b, "Mars is a planet.").status_code == 201
        second = upload(client, a, "Cancel subscriptions before renewal.", "terms.txt").json()["id"]
        retrieve = client.app.state.documents.retrieve
        assert any("Refunds" in hit["text"] for hit in retrieve(a, "refund", 5))
        assert retrieve(b, "refund", 5) == []
        assert client.delete(f"/api/v1/collections/{b}/documents/{second}").status_code == 404
        assert client.delete(f"/api/v1/collections/{a}/documents/{first.json()['id']}").status_code == 204
        assert [c["document_id"] for c in retrieve(a, "cancel", 5)] == [second]
    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        assert client.app.state.documents.retrieve(a, "cancel", 5)[0]["filename"] == "terms.txt"
        assert client.delete(f"/api/v1/collections/{a}/documents/{second}").status_code == 204
        assert client.app.state.documents.retrieve(a, "refund", 5) == []
        assert client.delete(f"/api/v1/collections/{b}").status_code == 204
        assert client.get(f"/api/v1/collections/{b}").status_code == 404
        assert list((settings.data_dir / "indexes").glob("*.faiss")) == []


def test_failed_index_write_preserves_collection(client, monkeypatch):
    cid = collection(client)
    assert upload(client, cid, "Refund policy.").status_code == 201
    service = client.app.state.documents
    def fail(_):
        raise OSError("Simulated disk failure")
    monkeypatch.setattr(service.store, "write", fail)
    assert upload(client, cid, "Cancel policy.").status_code == 500
    assert client.get(f"/api/v1/collections/{cid}").json()["document_count"] == 1
    assert service.retrieve(cid, "refund", 5)[0]["text"] == "Refund policy."


def test_invalid_upload_and_empty_collection(client):
    cid = collection(client)
    assert client.app.state.documents.retrieve(cid, "refund", 5) == []
    assert upload(client, cid, b"bad", "bad.pdf").status_code == 422
    assert upload(client, cid, b"bad", "bad.exe").status_code == 415
    assert upload(client, cid, b"x" * (10 * 1024 * 1024 + 1)).status_code == 413
    assert client.get(f"/api/v1/collections/{cid}").json()["chunk_count"] == 0


def test_configuration_change_and_missing_index(client):
    cid = collection(client)
    upload(client, cid, "Refunds are supported.")
    service = client.app.state.documents
    service.settings.chunk_tokens = 201
    with pytest.raises(AppError) as exc:
        service.retrieve(cid, "refund", 5)
    assert exc.value.code == "index_config_changed"
    service.settings.chunk_tokens = 200
    service.store.remove(service.repo.collection(cid)["snapshot"])
    with pytest.raises(AppError) as exc:
        service.retrieve(cid, "refund", 5)
    assert exc.value.code == "index_unavailable"


def test_orphan_cleanup(settings):
    index_dir = settings.data_dir / "indexes"
    index_dir.mkdir(parents=True)
    (index_dir / "orphan.faiss").write_bytes(b"incomplete")
    (index_dir / "interrupted.tmp").write_bytes(b"incomplete")
    with TestClient(create_app(settings, embedder=FakeEmbedder())):
        assert list(index_dir.iterdir()) == []


def test_busy_ingestion_returns_retryable_error(client):
    cid = collection(client)
    service = client.app.state.documents
    service.ingestion_slot.acquire()
    try:
        assert upload(client, cid, "Refund policy").status_code == 429
    finally:
        service.ingestion_slot.release()
