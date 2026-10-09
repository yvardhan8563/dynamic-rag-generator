import hashlib
import json
import logging
from threading import RLock, Semaphore
from uuid import uuid4

import numpy as np

from app.errors import AppError
from app.ingestion.chunking import chunk_passages
from app.ingestion.extractors import extract, safe_filename

logger = logging.getLogger("rag")


class DocumentService:
    def __init__(self, repo, store, embedder, settings):
        self.repo, self.store, self.embedder, self.settings = repo, store, embedder, settings
        self.locks = [RLock() for _ in range(64)]
        self.ingestion_slot = Semaphore(1)
        self.store.reconcile({c["snapshot"] for c in repo.collections() if c["snapshot"]})

    def lock(self, collection_id):
        return self.locks[int(hashlib.sha256(collection_id.encode()).hexdigest(), 16) % len(self.locks)]

    @property
    def fingerprint(self):
        return json.dumps({"model": self.settings.embedding_model, "tokens": self.settings.chunk_tokens,
                           "overlap": self.settings.chunk_overlap, "format": 1}, sort_keys=True)

    def check_config(self, collection):
        if collection["fingerprint"] and collection["fingerprint"] != self.fingerprint:
            raise AppError(409, "index_config_changed", "Embedding/chunk settings changed; restore settings or create a new collection and re-upload")

    def cleanup(self, snapshot):
        try:
            self.store.remove(snapshot)
        except OSError:
            logger.warning("Snapshot cleanup deferred until restart")

    def ingest(self, collection_id, filename, data):
        self.repo.collection(collection_id)
        filename = safe_filename(filename)
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise AppError(413, "upload_limit", "Document exceeds the configured upload limit")
        if not self.ingestion_slot.acquire(blocking=False):
            raise AppError(429, "ingestion_busy", "Another document is being indexed; retry shortly")
        try:
            content_hash = hashlib.sha256(data).hexdigest()
            with self.lock(collection_id):
                collection = self.repo.collection(collection_id)
                self.check_config(collection)
                if any(d["content_hash"] == content_hash for d in self.repo.documents(collection_id)):
                    raise AppError(409, "duplicate_document", "This document is already indexed in this collection")
            passages = extract(filename, data, self.settings.max_text_chars)
            size = min(self.settings.chunk_tokens, self.embedder.token_limit)
            if self.settings.chunk_overlap >= size:
                raise AppError(503, "invalid_chunk_settings", "Overlap is too large for the embedding model")
            chunks = chunk_passages(passages, self.embedder.tokenizer, size,
                                    self.settings.chunk_overlap, self.settings.max_chunks)
            vectors = self.embedder.documents([c["text"] for c in chunks])
            if len(vectors) != len(chunks):
                raise AppError(503, "invalid_embeddings", "Embedding count does not match chunks")
            document_id = str(uuid4())
            with self.lock(collection_id):
                collection = self.repo.collection(collection_id)
                old_chunks = self.repo.chunks(collection_id)
                if collection["snapshot"]:
                    old = self.store.read(collection["snapshot"], len(old_chunks))
                    if old.d != vectors.shape[1]:
                        raise AppError(409, "embedding_dimension_changed", "Embedding dimensions changed; re-upload into a new collection")
                    vectors = np.concatenate([old.reconstruct_n(0, old.ntotal), vectors])
                snapshot = self.store.write(vectors)
                try:
                    with self.repo.connect() as db:
                        db.execute("INSERT INTO documents(id,collection_id,filename,content_hash) VALUES (?,?,?,?)",
                                   (document_id, collection_id, filename, content_hash))
                        for position, chunk in enumerate(chunks, len(old_chunks)):
                            db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (chunk["id"], document_id,
                                       collection_id, chunk["text"], json.dumps(chunk["location"]), position))
                        db.execute("UPDATE collections SET snapshot=?, fingerprint=? WHERE id=?",
                                   (snapshot, self.fingerprint, collection_id))
                except Exception:
                    self.cleanup(snapshot)
                    raise
                self.cleanup(collection["snapshot"])
            return {"id": document_id, "collection_id": collection_id, "filename": filename,
                    "chunk_count": len(chunks), "status": "indexed"}
        finally:
            self.ingestion_slot.release()

    def delete_document(self, collection_id, document_id):
        with self.lock(collection_id):
            collection = self.repo.collection(collection_id)
            chunks = self.repo.chunks(collection_id)
            keep = [i for i, c in enumerate(chunks) if c["document_id"] != document_id]
            if len(keep) == len(chunks):
                raise AppError(404, "document_not_found", "Document not found in this collection")
            snapshot = None
            if keep:
                index = self.store.read(collection["snapshot"], len(chunks))
                snapshot = self.store.write(index.reconstruct_n(0, index.ntotal)[keep])
            try:
                with self.repo.connect() as db:
                    db.execute("DELETE FROM documents WHERE id=? AND collection_id=?", (document_id, collection_id))
                    for position, old in enumerate(keep):
                        db.execute("UPDATE chunks SET position=? WHERE id=?", (position, chunks[old]["id"]))
                    db.execute("UPDATE collections SET snapshot=?, fingerprint=? WHERE id=?",
                               (snapshot, collection["fingerprint"] if keep else None, collection_id))
            except Exception:
                self.cleanup(snapshot)
                raise
            self.cleanup(collection["snapshot"])

    def delete_collection(self, collection_id):
        with self.lock(collection_id):
            collection = self.repo.collection(collection_id)
            with self.repo.connect() as db:
                db.execute("DELETE FROM collections WHERE id=?", (collection_id,))
            self.cleanup(collection["snapshot"])

    def retrieve(self, collection_id, question, top_k):
        with self.lock(collection_id):
            collection = self.repo.collection(collection_id)
            self.check_config(collection)
            if not collection["snapshot"]:
                return []
        query = self.embedder.query(question)
        with self.lock(collection_id):
            collection = self.repo.collection(collection_id)
            chunks = self.repo.chunks(collection_id)
            if not chunks:
                return []
            index = self.store.read(collection["snapshot"], len(chunks))
            if index.d != query.shape[1]:
                raise AppError(409, "embedding_dimension_changed", "Embedding dimensions do not match the stored index")
            import faiss
            query = np.ascontiguousarray(query, dtype="float32")
            faiss.normalize_L2(query)
            scores, ids = index.search(query, min(top_k, len(chunks)))
            return [{**chunks[int(i)], "score": float(score)} for score, i in zip(scores[0], ids[0])
                    if i >= 0 and score >= self.settings.similarity_threshold]
