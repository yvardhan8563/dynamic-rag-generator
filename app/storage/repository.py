import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.errors import AppError


class Repository:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "metadata.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    snapshot TEXT, fingerprint TEXT
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL
                        REFERENCES collections(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL, content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(collection_id, content_hash)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                    text TEXT NOT NULL, location TEXT NOT NULL, position INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_collection ON chunks(collection_id, position);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_collection(self, name: str) -> dict:
        collection_id = str(uuid4())
        with self.connect() as db:
            db.execute("INSERT INTO collections(id,name) VALUES (?,?)", (collection_id, name))
        return self.collection(collection_id)

    def collection(self, collection_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("""SELECT c.*,
                (SELECT COUNT(*) FROM documents d WHERE d.collection_id=c.id) document_count,
                (SELECT COUNT(*) FROM chunks ch WHERE ch.collection_id=c.id) chunk_count
                FROM collections c WHERE c.id=?""", (collection_id,)).fetchone()
        if row is None:
            raise AppError(404, "collection_not_found", "Collection not found")
        return dict(row)

    def collections(self) -> list[dict]:
        with self.connect() as db:
            return [dict(row) for row in db.execute("""SELECT c.*,
                (SELECT COUNT(*) FROM documents d WHERE d.collection_id=c.id) document_count,
                (SELECT COUNT(*) FROM chunks ch WHERE ch.collection_id=c.id) chunk_count
                FROM collections c ORDER BY c.created_at,c.id""")]

    def documents(self, collection_id: str) -> list[dict]:
        self.collection(collection_id)
        with self.connect() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM documents WHERE collection_id=? ORDER BY created_at,id", (collection_id,))]

    def chunks(self, collection_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("""SELECT ch.*, d.filename FROM chunks ch
                JOIN documents d ON d.id=ch.document_id
                WHERE ch.collection_id=? ORDER BY ch.position""", (collection_id,)).fetchall()
        return [{**dict(r), "location": json.loads(r["location"])} for r in rows]
