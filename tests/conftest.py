import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.test_extraction import WordTokenizer


@pytest.fixture(autouse=True)
def isolate_provider_settings(monkeypatch):
    # Tests must never consume developer credentials or call a configured provider.
    for key in list(os.environ):
        if key.startswith("RAG_LLM_") or key in {"HF_TOKEN", "RAG_HF_TOKEN", "GEMINI_API_KEY", "GEMINI_MODEL"}:
            monkeypatch.delenv(key, raising=False)


class FakeEmbedder:
    """Deterministic test double; production always uses Sentence Transformers."""
    tokenizer = WordTokenizer()
    token_limit = 254
    loaded = True

    def documents(self, texts):
        result = []
        for text in texts:
            words = text.lower()
            vector = np.array([sum(words.count(w) for w in group) for group in
                               (("refund", "cancel"), ("planet", "mars"), ("recipe", "cook"))] + [0.01], dtype="float32")
            result.append(vector / np.linalg.norm(vector))
        return np.array(result, dtype="float32")

    def query(self, text):
        return self.documents([text])


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, _env_file=None)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, embedder=FakeEmbedder())) as client:
        yield client


def collection(client, name="Test"):
    return client.post("/api/v1/collections", json={"name": name}).json()["id"]


def upload(client, cid, text, filename="policy.txt"):
    return client.post(f"/api/v1/collections/{cid}/documents", files={"file": (filename, text, "text/plain")})
