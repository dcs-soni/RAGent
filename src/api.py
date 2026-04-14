"""
FastAPI Backend
================
Decouples the LangGraph execution from the Streamlit UI,
exposing a Server-Sent Events (SSE) streaming /chat endpoint.
"""

import hashlib
import hmac
import io
import logging
import json
import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import settings
from src.document_store import (
    create_document,
    delete_document,
    get_registry_summary,
    list_documents,
    update_document,
)
from src.ingestion import ingest_pipeline
from src.ingestion_jobs import job_tracker
from src.rate_limiter import RateLimit, rate_limiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if not hmac.compare_digest(credentials.credentials, settings.API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API Key",
        )
    return credentials.credentials


def _scope_thread_id(api_key: str, thread_id: str) -> str:
    """Namespace thread IDs by API key to prevent cross-tenant state leakage.

    Each tenant (identified by API key) gets an isolated conversation namespace.
    Even if two tenants use the same thread_id string, they will never collide
    because the key hash prefix makes them unique.
    """
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return f"{key_hash}:{thread_id}"

MAX_QUESTION_LENGTH = 2000  # characters
ALLOWED_EXTENSIONS = {".pdf"}
PDF_MAGIC_BYTES = b"%PDF-"

_graph = None
_ingestion_lock = Lock()
_ingestion_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once at startup instead of per-request."""
    global _graph
    try:
        from src.graph import build_graph
        _graph = build_graph()
        logger.info("✅ Graph compiled at startup")
    except Exception as e:
        logger.warning("⚠️ Graph not available at startup (missing vector DB?): %s", e)
        _graph = None
    yield
    _graph = None


_docs_enabled = os.getenv("DOCS_ENABLED", "false").lower() == "true"

app = FastAPI(
    title="RAGent API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def enforce_request_body_limits(request: Request, call_next):
    max_bytes = _request_body_limit_bytes(request)
    if max_bytes is not None:
        try:
            _validate_content_length(
                request,
                max_bytes,
                "Request body too large for this endpoint.",
            )
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )

    return await call_next(request)


@app.middleware("http")
async def set_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# Request Models 

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    thread_id: str = Field(
        default="default_session",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank")
        return cleaned


class IngestionStatusResponse(BaseModel):
    status: str
    stage: str
    message: str
    progress_percent: int
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error: str | None = None
    job_id: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


def _request_body_limit_bytes(request: Request) -> int | None:
    if request.method == "POST" and request.url.path in {"/documents", "/upload"}:
        return settings.max_upload_request_bytes
    if request.method == "PUT" and request.url.path.startswith("/documents/"):
        return settings.max_upload_request_bytes
    if request.method == "POST" and request.url.path == "/chat":
        return settings.max_chat_request_bytes
    return None


def _validate_content_length(request: Request, max_bytes: int, detail: str) -> None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        raise HTTPException(status_code=411, detail="Length Required: Content-Length header is mandatory")

    try:
        declared_size = int(content_length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc

    if declared_size > max_bytes:
        raise HTTPException(status_code=413, detail=detail)


def _client_identifier(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
        
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()

    client = request.client
    return client.host if client and client.host else "unknown"


def _enforce_rate_limit(
    request: Request,
    scope: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    retry_after = rate_limiter.check(
        scope=scope,
        identifier=_client_identifier(request),
        limit=RateLimit(requests=max_requests, window_seconds=window_seconds),
    )
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )


def limit_chat_requests(request: Request) -> None:
    _enforce_rate_limit(
        request=request,
        scope="chat",
        max_requests=settings.CHAT_RATE_LIMIT_REQUESTS,
        window_seconds=settings.CHAT_RATE_LIMIT_WINDOW_SECONDS,
    )


def limit_document_write_requests(request: Request) -> None:
    _enforce_rate_limit(
        request=request,
        scope="document_write",
        max_requests=settings.DOCUMENT_WRITE_RATE_LIMIT_REQUESTS,
        window_seconds=settings.DOCUMENT_WRITE_RATE_LIMIT_WINDOW_SECONDS,
    )


def limit_ingest_requests(request: Request) -> None:
    _enforce_rate_limit(
        request=request,
        scope="ingest",
        max_requests=settings.INGEST_RATE_LIMIT_REQUESTS,
        window_seconds=settings.INGEST_RATE_LIMIT_WINDOW_SECONDS,
    )


def _validate_pdf_upload_name(file: UploadFile) -> str:
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if len(safe_name) > settings.MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Filename is too long")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed",
        )
    return safe_name


def _validate_pdf_content_safely(content: bytes) -> None:
    if PDF_MAGIC_BYTES not in content[:1024]:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")
        
    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
        if len(reader.pages) == 0:
            raise HTTPException(status_code=400, detail="PDF has no pages")
    except PdfReadError as e:
        logger.warning("Malformed or encrypted PDF upload attempt: %s", e)
        raise HTTPException(status_code=400, detail="Malformed or corrupted PDF file")
    except Exception:
        logger.exception("Failed to parse PDF during validation")
        raise HTTPException(status_code=500, detail="Error validating PDF file")


async def _read_upload_content(request: Request, file: UploadFile) -> tuple[str, bytes]:
    _validate_content_length(
        request,
        settings.max_upload_request_bytes,
        f"Request body too large. Maximum upload size is {settings.MAX_UPLOAD_SIZE_MB} MB.",
    )

    safe_name = _validate_pdf_upload_name(file)
    content = bytearray()

    try:
        while True:
            chunk = await file.read(settings.upload_read_chunk_size_bytes)
            if not chunk:
                break

            content.extend(chunk)
            if len(content) > settings.max_upload_size_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB",
                )
    finally:
        await file.close()

    _validate_pdf_content_safely(bytes(content))
    return safe_name, bytes(content)


def _latest_job_response() -> IngestionStatusResponse:
    latest_job = job_tracker.get_latest_job()
    if latest_job is None:
        return IngestionStatusResponse(
            status="idle",
            stage="idle",
            message="No ingestion job has been started yet",
            progress_percent=0,
        )
    return IngestionStatusResponse(**latest_job.to_response())


def _ensure_document_mutations_allowed() -> None:
    if _ingestion_running:
        raise HTTPException(
            status_code=409,
            detail="Document changes are disabled while ingestion is running",
        )


#  Endpoints 

@app.get("/health")
def health_check():
    registry = get_registry_summary()
    status = "ok"
    if registry.active_documents == 0:
        status = "no_documents"
    elif not settings.chroma_path.exists() or registry.indexed_content_version == 0:
        status = "no_vector_db"
    elif registry.needs_reindex:
        status = "stale_index"
    return {
        "status": status,
        "graph_ready": _graph is not None and not registry.needs_reindex,
    }

@app.get("/system/status", dependencies=[Depends(verify_api_key)])
def system_status_endpoint():
    registry = get_registry_summary()
    status = "ok"
    if registry.active_documents == 0:
        status = "no_documents"
    elif not settings.chroma_path.exists() or registry.indexed_content_version == 0:
        status = "no_vector_db"
    elif registry.needs_reindex:
        status = "stale_index"
    return {
        "status": status,
        "graph_ready": _graph is not None and not registry.needs_reindex,
        "documents": registry.model_dump(mode="json"),
        "ingestion": _latest_job_response().model_dump(mode="json"),
    }


@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents_endpoint():
    registry = get_registry_summary()
    return {
        "documents": [document.model_dump(mode="json") for document in list_documents()],
        "summary": registry.model_dump(mode="json"),
    }


@app.post(
    "/documents",
    dependencies=[Depends(verify_api_key), Depends(limit_document_write_requests)],
)
async def create_document_endpoint(request: Request, file: UploadFile = File(...)):
    """Create a new managed document with a stable ID."""
    try:
        _ensure_document_mutations_allowed()
        safe_name, content = await _read_upload_content(request, file)
        record = create_document(
            original_filename=safe_name,
            content=content,
            content_type=file.content_type or "application/pdf",
        )
        return {"document": record.model_dump(mode="json")}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Document upload failed")


@app.post(
    "/upload",
    dependencies=[Depends(verify_api_key), Depends(limit_document_write_requests)],
)
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Backward-compatible alias for document creation."""
    return await create_document_endpoint(request, file)


@app.put(
    "/documents/{document_id}",
    dependencies=[Depends(verify_api_key), Depends(limit_document_write_requests)],
)
async def update_document_endpoint(document_id: str, request: Request, file: UploadFile = File(...)):
    """Replace an existing document while preserving its stable ID."""
    try:
        _ensure_document_mutations_allowed()
        safe_name, content = await _read_upload_content(request, file)
        record = update_document(
            document_id=document_id,
            original_filename=safe_name,
            content=content,
            content_type=file.content_type or "application/pdf",
        )
        return {"document": record.model_dump(mode="json")}
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document update failed")
        raise HTTPException(status_code=500, detail="Document update failed")


@app.delete(
    "/documents/{document_id}",
    dependencies=[Depends(verify_api_key), Depends(limit_document_write_requests)],
)
def delete_document_endpoint(document_id: str):
    """Delete an existing document and mark the index as stale."""
    try:
        _ensure_document_mutations_allowed()
        record = delete_document(document_id)
        return {"document": record.model_dump(mode="json")}
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        logger.exception("Document delete failed")
        raise HTTPException(status_code=500, detail="Document delete failed")


@app.post("/ingest", dependencies=[Depends(verify_api_key), Depends(limit_ingest_requests)])
async def ingest_endpoint(background_tasks: BackgroundTasks):
    """Triggers the ingestion pipeline in the background."""
    global _graph, _ingestion_running
    try:
        registry = get_registry_summary()
        if registry.active_documents == 0:
            raise HTTPException(
                status_code=400,
                detail="No active documents found. Upload documents first.",
            )

        with _ingestion_lock:
            if _ingestion_running:
                raise HTTPException(status_code=409, detail="Ingestion is already running")
            _ingestion_running = True

        job = job_tracker.create_job(total_documents=registry.active_documents)

        def _ingest_and_rebuild():
            """Run ingestion, then rebuild the graph so new vectors are available."""
            global _graph, _ingestion_running
            try:
                job_tracker.start_job(job.job_id)

                def _progress_update(payload: dict) -> None:
                    job_tracker.update_job(
                        job.job_id,
                        status="running",
                        stage=str(payload.get("stage", "running")),
                        message=str(payload.get("message", "Ingestion running")),
                        total_documents=int(payload.get("total_documents", registry.active_documents)),
                        processed_documents=int(payload.get("processed_documents", 0)),
                        total_chunks=int(payload.get("total_chunks", 0)),
                        processed_chunks=int(payload.get("processed_chunks", 0)),
                    )

                ingest_pipeline(progress_callback=_progress_update)
                from src.graph import build_graph
                from src.nodes import reset_retriever_cache
                reset_retriever_cache()
                _graph = build_graph()
                job_tracker.finish_job(job.job_id, "Ingestion completed successfully")
                logger.info("✅ Graph rebuilt after ingestion")
            except Exception as exc:
                job_tracker.fail_job(job.job_id, str(exc))
                logger.exception("Background ingestion failed")
            finally:
                with _ingestion_lock:
                    _ingestion_running = False

        background_tasks.add_task(_ingest_and_rebuild)
        return {
            "status": "Ingestion started in the background",
            "job": job_tracker.get_job(job.job_id).to_response(),
        }
    except HTTPException:
        with _ingestion_lock:
            _ingestion_running = False
        raise
    except Exception as e:
        logger.exception("Ingest endpoint failed")
        with _ingestion_lock:
            _ingestion_running = False
        raise HTTPException(status_code=500, detail="Ingestion trigger failed")


@app.get("/ingest/status", dependencies=[Depends(verify_api_key)])
def ingest_status_endpoint():
    return _latest_job_response().model_dump(mode="json")


@app.get("/ingest/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def ingest_job_endpoint(job_id: str):
    job = job_tracker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job.to_response()


@app.post("/chat", dependencies=[Depends(limit_chat_requests)])
async def chat_endpoint(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    """Stream state updates using LangGraph async stream via SSE."""
    registry = get_registry_summary()
    if registry.active_documents == 0:
        raise HTTPException(
            status_code=400,
            detail="No active documents available. Upload and ingest documents first.",
        )
    if registry.needs_reindex:
        raise HTTPException(
            status_code=409,
            detail="Documents changed since the last ingestion. Run ingestion before chatting.",
        )
    if _graph is None:
        raise HTTPException(
            status_code=503,
            detail="Agent graph not available. Run document ingestion first.",
        )

    # Scope thread_id by API key to prevent cross-tenant state leakage (SEC-03)
    scoped_thread_id = _scope_thread_id(api_key, request.thread_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "question": request.question,
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "retry_count": 0,
        }

        try:
            config = {"configurable": {"thread_id": scoped_thread_id}}
            async for output in _graph.astream(initial_state, config=config):
                for node_name, state_update in output.items():
                    event_data = {
                        "node": node_name,
                        "generation": state_update.get("generation", ""),
                        "retry_count": state_update.get("retry_count", 0),
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("Error in graph execution")
            yield f"data: {json.dumps({'error': 'Agent execution failed'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
