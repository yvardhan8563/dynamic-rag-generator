import os
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from app.errors import AppError


class FaissStore:
    def __init__(self, directory: Path):
        self.directory = directory / "indexes"
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        # Names are generated internally, never accepted from upload filenames.
        if Path(name).name != name or not name.endswith(".faiss"):
            raise AppError(503, "invalid_index", "Invalid stored index reference")
        return self.directory / name

    def write(self, vectors: np.ndarray) -> str:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise AppError(503, "invalid_embeddings", "Embedding model produced invalid vectors")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        name = f"{uuid4()}.faiss"
        destination = self.path(name)
        temporary = destination.with_suffix(".tmp")
        try:
            # Python file I/O handles Unicode Windows paths reliably.
            with temporary.open("wb") as stream:
                stream.write(faiss.serialize_index(index).tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return name

    def read(self, name: str, count: int):
        try:
            index = faiss.deserialize_index(np.frombuffer(self.path(name).read_bytes(), dtype="uint8"))
            if index.ntotal != count or not isinstance(index, faiss.IndexFlatIP):
                raise ValueError("Index mismatch")
            return index
        except Exception as exc:
            raise AppError(503, "index_unavailable", "Stored index is missing or inconsistent; restore data or re-upload documents") from exc

    def remove(self, name: str | None):
        if name:
            self.path(name).unlink(missing_ok=True)

    def reconcile(self, active: set[str]):
        for path in self.directory.iterdir():
            if path.suffix in {".faiss", ".tmp"} and path.name not in active:
                path.unlink()
