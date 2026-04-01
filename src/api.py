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
from src.ingestion import ingest_pipeline

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


#  Endpoints 

@app.get("/health")
def health_check():
    status = "ok"
    if not settings.chroma_path.exists():
        status = "no_vector_db"
    return {"status": status, "graph_ready": _graph is not None}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a PDF to the documents directory with security checks."""
    # Sanitize filename — strip path components to prevent traversal attacks
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if len(safe_name) > settings.MAX_FILENAME_LENGTH:
        raise HTTPException(status_code=400, detail="Filename is too long")

    # Validate extension
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed",
        )

    try:
        settings.docs_path.mkdir(parents=True, exist_ok=True)
        file_path = settings.docs_path / safe_name
        if file_path.exists():
            raise HTTPException(
                status_code=409,
                detail=f"A document named '{safe_name}' already exists",
            )

        # Read with size limit to prevent resource exhaustion
        content = await file.read()
        if len(content) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB",
            )
        if PDF_MAGIC_BYTES not in content[:1024]:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")

        with open(file_path, "wb") as f:
            f.write(content)

        return {"filename": safe_name, "status": "uploaded", "size_bytes": len(content)}
    except HTTPException:
        raise  # Re-raise our own HTTPExceptions
    except Exception as e:
        logger.exception("Upload failed for file: %s", safe_name)
        raise HTTPException(status_code=500, detail="Upload failed unexpectedly")


@app.post("/ingest")
async def ingest_endpoint(background_tasks: BackgroundTasks):
    """Triggers the ingestion pipeline in the background."""
    global _graph, _ingestion_running
    try:
        with _ingestion_lock:
            if _ingestion_running:
                raise HTTPException(status_code=409, detail="Ingestion is already running")
            _ingestion_running = True

        if not settings.docs_path.exists():
            settings.docs_path.mkdir(parents=True, exist_ok=True)

        pdf_files = list(settings.docs_path.glob("*.pdf"))
        if not pdf_files:
            raise HTTPException(
                status_code=400,
                detail="No PDF files found in the documents directory. Upload files first.",
            )

        def _ingest_and_rebuild():
            """Run ingestion, then rebuild the graph so new vectors are available."""
            global _graph, _ingestion_running
            try:
                ingest_pipeline()
                from src.graph import build_graph
                from src.nodes import reset_retriever_cache
                reset_retriever_cache()
                _graph = build_graph()
                logger.info("✅ Graph rebuilt after ingestion")
            except Exception:
                logger.exception("Background ingestion failed")
            finally:
                with _ingestion_lock:
                    _ingestion_running = False

        background_tasks.add_task(_ingest_and_rebuild)
        return {"status": "Ingestion started in the background", "files": len(pdf_files)}
    except HTTPException:
        with _ingestion_lock:
            _ingestion_running = False
        raise
    except Exception as e:
        logger.exception("Ingest endpoint failed")
        with _ingestion_lock:
            _ingestion_running = False
        raise HTTPException(status_code=500, detail="Ingestion trigger failed")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Stream state updates using LangGraph async stream via SSE."""
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
