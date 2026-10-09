from concurrent.futures import ThreadPoolExecutor

from tests.conftest import collection, upload


def test_database_failure_discards_uncommitted_snapshot(client):
    cid = collection(client)
    upload(client, cid, "Refund policy.")
    service = client.app.state.documents
    original = service.repo.collection(cid)["snapshot"]
    with service.repo.connect() as db:
        db.execute("""CREATE TRIGGER reject_test_insert BEFORE INSERT ON chunks
                      BEGIN SELECT RAISE(ABORT, 'Simulated database failure'); END""")
    assert upload(client, cid, "Cancel subscription.").status_code == 500
    assert service.repo.collection(cid)["snapshot"] == original
    assert len(service.repo.documents(cid)) == 1
    assert [p.name for p in service.store.directory.glob("*.faiss")] == [original]
    assert service.retrieve(cid, "refund", 5)[0]["text"] == "Refund policy."


def test_concurrent_queries_and_document_deletion(client):
    cid = collection(client)
    doc = upload(client, cid, "Refund policy.").json()["id"]
    service = client.app.state.documents
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(service.retrieve, cid, "refund", 5) for _ in range(10)]
        deleted = executor.submit(service.delete_document, cid, doc)
        for future in futures:
            hits = future.result()
            assert hits == [] or hits[0]["document_id"] == doc
        deleted.result()
    assert service.retrieve(cid, "refund", 5) == []


def test_chunked_request_limit(client):
    cid = collection(client)
    def content():
        for _ in range(12):
            yield b"x" * 1024 * 1024
    response = client.post(f"/api/v1/collections/{cid}/documents", content=content())
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_limit"
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]
