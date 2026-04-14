"""
FastAPI Backend
================
Decouples the LangGraph execution from the Streamlit UI,
exposing a Server-Sent Events (SSE) streaming /chat endpoint.
"""

import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException, Depends, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator

from src.config import settings
from src.document_store import get_registry_summary, list_documents
from src.ingestion_jobs import job_tracker
from src.rate_limiter import RateLimit, rate_limiter

from src.services.document_service import (
    process_and_create_document,
    process_and_update_document,
    process_and_delete_document,
)
from src.services.ingestion_service import (
    get_latest_job_response,
    trigger_background_ingestion,
)
from src.services.chat_service import (
    load_graph,
    unload_graph,
    is_graph_ready,
    scope_thread_id,
    generate_chat_stream,
)

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


MAX_QUESTION_LENGTH = 2000

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the graph once at startup instead of per-request."""
    load_graph()
    yield
    unload_graph()


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

def _request_body_limit_bytes(request: Request) -> int | None:
    if request.method == "POST" and request.url.path in {"/documents", "/upload"}:
        return settings.max_upload_request_bytes
    if request.method == "PUT" and request.url.path.startswith("/documents/"):
        return settings.max_upload_request_bytes
    if request.method == "POST" and request.url.path == "/chat":
        return settings.max_chat_request_bytes
    return None

@app.middleware("http")
async def enforce_request_body_limits(request: Request, call_next):
    max_bytes = _request_body_limit_bytes(request)
    if max_bytes is not None:
        content_length = request.headers.get("content-length")
        if content_length is None:
            return JSONResponse(status_code=411, content={"detail": "Length Required: Content-Length header is mandatory"})
        try:
            declared_size = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})
        
        if declared_size > max_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request body too large for this endpoint."})

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


# Endpoints 

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
        "graph_ready": is_graph_ready() and not registry.needs_reindex,
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
        "graph_ready": is_graph_ready() and not registry.needs_reindex,
        "documents": registry.model_dump(mode="json"),
        "ingestion": get_latest_job_response(),
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
        record = await process_and_create_document(request, file)
        return {"document": record.model_dump(mode="json")}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BufferError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
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
        record = await process_and_update_document(document_id, request, file)
        return {"document": record.model_dump(mode="json")}
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except BufferError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
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
        record = process_and_delete_document(document_id)
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
    try:
        return trigger_background_ingestion(background_tasks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        if str(exc) == "Ingestion is already running":
            raise HTTPException(status_code=409, detail=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception:
        logger.exception("Ingest endpoint failed")
        raise HTTPException(status_code=500, detail="Ingestion trigger failed")


@app.get("/ingest/status", dependencies=[Depends(verify_api_key)])
def ingest_status_endpoint():
    return get_latest_job_response()


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
    scoped_thread = scope_thread_id(api_key, request.thread_id)

    return StreamingResponse(
        generate_chat_stream(request.question, scoped_thread),
        media_type="text/event-stream"
    )
