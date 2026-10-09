import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from filelock import FileLock, Timeout

from app.config import Settings
from app.errors import AppError
from app.schemas import CollectionCreate, Query, CollectionRead, DocumentRead, UploadResult, Answer
from app.storage.repository import Repository
from app.retrieval.embeddings import Embedder
from app.retrieval.faiss_store import FaissStore
from app.services.ingestion import DocumentService
from app.services.answering import AnswerService
from app.generation.client import LLMClient
from app.body_limit import BodyLimitMiddleware

logger = logging.getLogger("rag")


def create_app(settings: Settings | None = None, *, embedder=None, llm=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(settings.data_dir / "server.lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("This data directory is already used by another backend. Run one worker.") from exc
        try:
            app.state.repo = Repository(settings.data_dir)
            app.state.embedder = embedder or Embedder(settings.embedding_model)
            app.state.documents = DocumentService(app.state.repo, FaissStore(settings.data_dir),
                                                   app.state.embedder, settings)
            app.state.answers = AnswerService(app.state.documents, llm or LLMClient(settings), settings)
            yield
        finally:
            lock.release()

    app = FastAPI(title="Dynamic RAG Generator", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_mb * 1024 * 1024 + 64 * 1024)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request.state.request_id = str(uuid4())
        try:
            response = await call_next(request)
        except Exception:
            # Do not log exception strings: providers and parsers can include sensitive text.
            logger.error("Unhandled request failure request_id=%s", request.state.request_id)
            response = JSONResponse(status_code=500, content={"error": {
                "code": "internal_error", "message": "An internal error occurred",
                "request_id": request.state.request_id}})
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(AppError)
    async def app_error(request: Request, exc: AppError):
        return JSONResponse(status_code=exc.status, content={"error": {
            "code": exc.code, "message": exc.message, "request_id": request.state.request_id}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {
            "code": "validation_error", "message": "Invalid request fields",
            "request_id": request.state.request_id}})

    @app.get("/api/v1/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/api/v1/health/ready")
    def ready(request: Request):
        with request.app.state.repo.connect() as db:
            db.execute("SELECT 1")
        configured = settings.llm_configured
        return JSONResponse(status_code=200 if configured else 503, content={
            "storage": "ready", "embeddings": "loaded" if request.app.state.embedder.loaded else "not_loaded",
            "llm": "configured_not_probed" if configured else "unconfigured",
            "provider": "gemini" if settings.is_gemini else ("huggingface" if settings.is_huggingface else "compatible"),
            "model": settings.generation_model})

    @app.post("/api/v1/collections", status_code=201, response_model=CollectionRead)
    def create_collection(body: CollectionCreate, request: Request):
        return request.app.state.repo.create_collection(body.name)

    @app.get("/api/v1/collections", response_model=list[CollectionRead])
    def collections(request: Request):
        return request.app.state.repo.collections()

    @app.get("/api/v1/collections/{collection_id}", response_model=CollectionRead)
    def collection(collection_id: UUID, request: Request):
        return request.app.state.repo.collection(str(collection_id))

    @app.delete("/api/v1/collections/{collection_id}", status_code=204)
    def delete_collection(collection_id: UUID, request: Request):
        request.app.state.documents.delete_collection(str(collection_id))
        return Response(status_code=204)

    @app.post("/api/v1/collections/{collection_id}/documents", status_code=201, response_model=UploadResult)
    def upload(collection_id: UUID, request: Request, file: UploadFile = File(...)):
        try:
            data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
            return request.app.state.documents.ingest(str(collection_id), file.filename or "", data)
        finally:
            file.file.close()

    @app.get("/api/v1/collections/{collection_id}/documents", response_model=list[DocumentRead])
    def documents(collection_id: UUID, request: Request):
        with request.app.state.documents.lock(str(collection_id)):
            return request.app.state.repo.documents(str(collection_id))

    @app.delete("/api/v1/collections/{collection_id}/documents/{document_id}", status_code=204)
    def delete_document(collection_id: UUID, document_id: UUID, request: Request):
        request.app.state.documents.delete_document(str(collection_id), str(document_id))
        return Response(status_code=204)

    @app.post("/api/v1/collections/{collection_id}/query", response_model=Answer)
    def query(collection_id: UUID, body: Query, request: Request):
        return request.app.state.answers.answer(str(collection_id), body.question, body.top_k,
                                                request.state.request_id)

    return app


app = create_app()
