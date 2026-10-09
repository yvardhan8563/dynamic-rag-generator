import os

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import collection, upload


@pytest.mark.model
@pytest.mark.skipif(os.getenv("RAG_RUN_MODEL_TESTS") != "1", reason="Set RAG_RUN_MODEL_TESTS=1 for real model checks")
def test_real_sentence_transformer_retrieval(settings):
    with TestClient(create_app(settings)) as client:
        cid = collection(client)
        assert upload(client, cid, "Customers may return unused items within thirty days for a full refund.").status_code == 201
        assert upload(client, cid, "Mars is the fourth planet from the Sun.", "space.txt").status_code == 201
        hits = client.app.state.documents.retrieve(cid, "How long do I have to return a purchase?", 2)
        assert hits and hits[0]["filename"] == "policy.txt"
        assert client.app.state.embedder.loaded
