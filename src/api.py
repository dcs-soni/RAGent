"""
FastAPI Backend
================
Decouples the LangGraph execution from the Streamlit UI,
exposing a Server-Sent Events (SSE) streaming /chat endpoint.
"""

import logging
import json
from contextlib import asynccontextmanager
from threading import Lock
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


app = FastAPI(
    title="RAGent API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def set_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
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


def _validate_pdf_upload(file: UploadFile, content: bytes) -> str:
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
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB",
        )
    if PDF_MAGIC_BYTES not in content[:1024]:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")
    return safe_name


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
        "documents": registry.model_dump(mode="json"),
        "ingestion": _latest_job_response().model_dump(mode="json"),
    }


@app.get("/documents")
def list_documents_endpoint():
    registry = get_registry_summary()
    return {
        "documents": [document.model_dump(mode="json") for document in list_documents()],
        "summary": registry.model_dump(mode="json"),
    }


@app.post("/documents")
async def create_document_endpoint(file: UploadFile = File(...)):
    """Create a new managed document with a stable ID."""
    try:
        _ensure_document_mutations_allowed()
        content = await file.read()
        safe_name = _validate_pdf_upload(file, content)
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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Backward-compatible alias for document creation."""
    return await create_document_endpoint(file)


@app.put("/documents/{document_id}")
async def update_document_endpoint(document_id: str, file: UploadFile = File(...)):
    """Replace an existing document while preserving its stable ID."""
    try:
        _ensure_document_mutations_allowed()
        content = await file.read()
        safe_name = _validate_pdf_upload(file, content)
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


@app.delete("/documents/{document_id}")
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


@app.post("/ingest")
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


@app.get("/ingest/status")
def ingest_status_endpoint():
    return _latest_job_response().model_dump(mode="json")


@app.get("/ingest/jobs/{job_id}")
def ingest_job_endpoint(job_id: str):
    job = job_tracker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job.to_response()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
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

    async def event_generator() -> AsyncGenerator[str, None]:
        initial_state = {
            "question": request.question,
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "retry_count": 0,
        }

        try:
            config = {"configurable": {"thread_id": request.thread_id}}
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
