from threading import Lock

import numpy as np

from app.errors import AppError


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    @property
    def loaded(self):
        return self._model is not None

    def load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.model_name, device="cpu", trust_remote_code=False)
                except Exception as exc:
                    raise AppError(503, "embedding_unavailable", "Embedding model could not be loaded; check installation and model cache") from exc
        return self._model

    @property
    def tokenizer(self):
        return self.load().tokenizer

    @property
    def token_limit(self):
        model = self.load()
        return model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)

    def documents(self, texts: list[str]) -> np.ndarray:
        model = self.load()
        try:
            with self._lock:
                return np.asarray(model.encode_document(texts, normalize_embeddings=True,
                                  batch_size=32, show_progress_bar=False), dtype="float32")
        except AppError:
            raise
        except Exception as exc:
            raise AppError(503, "embedding_failed", "Document embedding failed") from exc

    def query(self, text: str) -> np.ndarray:
        model = self.load()
        try:
            if len(model.tokenizer.encode(text, add_special_tokens=True)) > model.max_seq_length:
                raise AppError(422, "question_too_long", "Question exceeds the embedding model token limit; shorten it")
            with self._lock:
                return np.asarray(model.encode_query([text], normalize_embeddings=True,
                                  show_progress_bar=False), dtype="float32")
        except AppError:
            raise
        except Exception as exc:
            raise AppError(503, "embedding_failed", "Query embedding failed") from exc
