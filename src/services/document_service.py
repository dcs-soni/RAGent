import io
import logging
from pathlib import Path
from fastapi import Request, UploadFile

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.config import settings
from src.document_store import create_document, update_document, delete_document
from src.services.ingestion_service import assert_mutations_allowed

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf"}
PDF_MAGIC_BYTES = b"%PDF-"

def _validate_content_length(request: Request, max_bytes: int, detail: str) -> None:
    content_length = request.headers.get("content-length")
    if content_length is None:
        raise ValueError("Length Required: Content-Length header is mandatory")

    try:
        declared_size = int(content_length)
    except ValueError as exc:
        raise ValueError("Invalid Content-Length header") from exc

    if declared_size > max_bytes:
        raise BufferError(detail)

def _validate_pdf_upload_name(file: UploadFile) -> str:
    safe_name = Path(file.filename or "").name
    if not safe_name:
        raise ValueError("Invalid filename")
    if len(safe_name) > settings.MAX_FILENAME_LENGTH:
        raise ValueError("Filename is too long")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed")
    return safe_name

def _validate_pdf_content_safely(content: bytes) -> None:
    if PDF_MAGIC_BYTES not in content[:1024]:
        raise ValueError("Uploaded file is not a valid PDF")
        
    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
        if len(reader.pages) == 0:
            raise ValueError("PDF has no pages")
    except PdfReadError as e:
        logger.warning("Malformed or encrypted PDF upload attempt: %s", e)
        raise ValueError("Malformed or corrupted PDF file")
    except Exception:
        logger.exception("Failed to parse PDF during validation")
        raise RuntimeError("Error validating PDF file")

async def read_upload_content(request: Request, file: UploadFile) -> tuple[str, bytes]:
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
                raise BufferError(f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE_MB} MB")
    finally:
        await file.close()

    _validate_pdf_content_safely(bytes(content))
    return safe_name, bytes(content)


async def process_and_create_document(request: Request, file: UploadFile):
    assert_mutations_allowed()
    safe_name, content = await read_upload_content(request, file)
    return create_document(
        original_filename=safe_name,
        content=content,
        content_type=file.content_type or "application/pdf",
    )

async def process_and_update_document(document_id: str, request: Request, file: UploadFile):
    assert_mutations_allowed()
    safe_name, content = await read_upload_content(request, file)
    return update_document(
        document_id=document_id,
        original_filename=safe_name,
        content=content,
        content_type=file.content_type or "application/pdf",
    )

def process_and_delete_document(document_id: str):
    assert_mutations_allowed()
    return delete_document(document_id)
