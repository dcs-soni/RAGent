from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from threading import Lock

from pydantic import BaseModel, Field

from src.config import settings
from src.utils import utc_now_iso


_registry_lock = Lock()

class DocumentRecord(BaseModel):
    document_id: str
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    version: int = 1
    status: str = "active"
    created_at: str
    updated_at: str
    deleted_at: str | None = None
    last_ingested_at: str | None = None


class DocumentRegistry(BaseModel):
    version: int = 1
    content_version: int = 0
    indexed_content_version: int = 0
    documents: list[DocumentRecord] = Field(default_factory=list)


class RegistrySummary(BaseModel):
    total_documents: int
    active_documents: int
    deleted_documents: int
    content_version: int
    indexed_content_version: int
    needs_reindex: bool


def _ensure_docs_dir() -> None:
    settings.docs_path.mkdir(parents=True, exist_ok=True)


def _registry_path() -> Path:
    _ensure_docs_dir()
    return settings.document_registry_path


def _load_registry_unlocked() -> DocumentRegistry:
    path = _registry_path()
    if not path.exists():
        return DocumentRegistry()
    data = json.loads(path.read_text(encoding="utf-8"))
    return DocumentRegistry.model_validate(data)


def _save_registry_unlocked(registry: DocumentRegistry) -> None:
    path = _registry_path()
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(registry.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _build_storage_name(document_id: str, version: int, suffix: str) -> str:
    normalized_suffix = suffix.lower() or ".pdf"
    return f"{document_id}_v{version}{normalized_suffix}"


def _write_document_file(stored_filename: str, content: bytes) -> Path:
    target_path = settings.docs_path / stored_filename
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    temp_path.write_bytes(content)
    temp_path.replace(target_path)
    return target_path


def list_documents(include_deleted: bool = False) -> list[DocumentRecord]:
    with _registry_lock:
        registry = _load_registry_unlocked()
        documents = registry.documents
        if not include_deleted:
            documents = [doc for doc in documents if doc.status == "active"]
        return sorted(documents, key=lambda doc: doc.updated_at, reverse=True)


def get_document(document_id: str, include_deleted: bool = False) -> DocumentRecord | None:
    documents = list_documents(include_deleted=include_deleted)
    for document in documents:
        if document.document_id == document_id:
            return document
    return None


def get_registry_summary() -> RegistrySummary:
    with _registry_lock:
        registry = _load_registry_unlocked()
        total = len(registry.documents)
        active = sum(1 for doc in registry.documents if doc.status == "active")
        deleted = total - active
        return RegistrySummary(
            total_documents=total,
            active_documents=active,
            deleted_documents=deleted,
            content_version=registry.content_version,
            indexed_content_version=registry.indexed_content_version,
            needs_reindex=registry.content_version != registry.indexed_content_version,
        )


def create_document(
    original_filename: str,
    content: bytes,
    content_type: str = "application/pdf",
) -> DocumentRecord:
    suffix = Path(original_filename).suffix or ".pdf"
    now = utc_now_iso()
    document_id = uuid.uuid4().hex
    stored_filename = _build_storage_name(document_id, 1, suffix)
    record = DocumentRecord(
        document_id=document_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=len(content),
        sha256=_compute_sha256(content),
        version=1,
        created_at=now,
        updated_at=now,
    )

    with _registry_lock:
        registry = _load_registry_unlocked()
        try:
            _write_document_file(stored_filename, content)
            registry.documents.append(record)
            registry.content_version += 1
            _save_registry_unlocked(registry)
        except Exception:
            try:
                os.remove(settings.docs_path / stored_filename)
            except OSError:
                pass
            raise
    return record


def update_document(
    document_id: str,
    original_filename: str,
    content: bytes,
    content_type: str = "application/pdf",
) -> DocumentRecord:
    suffix = Path(original_filename).suffix or ".pdf"
    now = utc_now_iso()

    with _registry_lock:
        registry = _load_registry_unlocked()
        for index, record in enumerate(registry.documents):
            if record.document_id != document_id:
                continue
            if record.status != "active":
                raise ValueError("Document is deleted and cannot be updated")

            next_version = record.version + 1
            stored_filename = _build_storage_name(document_id, next_version, suffix)
            updated_record = record.model_copy(
                update={
                    "original_filename": original_filename,
                    "stored_filename": stored_filename,
                    "content_type": content_type,
                    "size_bytes": len(content),
                    "sha256": _compute_sha256(content),
                    "version": next_version,
                    "updated_at": now,
                    "last_ingested_at": None,
                }
            )

            old_path = settings.docs_path / record.stored_filename
            new_path = settings.docs_path / stored_filename

            try:
                _write_document_file(stored_filename, content)
                registry.documents[index] = updated_record
                registry.content_version += 1
                _save_registry_unlocked(registry)
            except Exception:
                try:
                    os.remove(new_path)
                except OSError:
                    pass
                raise

            if old_path.exists():
                old_path.unlink()
            return updated_record

    raise KeyError(f"Document '{document_id}' not found")


def delete_document(document_id: str) -> DocumentRecord:
    now = utc_now_iso()

    with _registry_lock:
        registry = _load_registry_unlocked()
        for index, record in enumerate(registry.documents):
            if record.document_id != document_id:
                continue
            if record.status != "active":
                raise ValueError("Document is already deleted")

            updated_record = record.model_copy(
                update={
                    "status": "deleted",
                    "deleted_at": now,
                    "updated_at": now,
                    "last_ingested_at": None,
                }
            )
            registry.documents[index] = updated_record
            registry.content_version += 1
            _save_registry_unlocked(registry)

            file_path = settings.docs_path / record.stored_filename
            if file_path.exists():
                file_path.unlink()
            return updated_record

    raise KeyError(f"Document '{document_id}' not found")


def get_active_document_paths() -> list[tuple[DocumentRecord, Path]]:
    documents = list_documents(include_deleted=False)
    paths: list[tuple[DocumentRecord, Path]] = []
    for document in documents:
        file_path = settings.docs_path / document.stored_filename
        if file_path.exists():
            paths.append((document, file_path))
    return paths


def mark_indexed(document_ids: list[str]) -> None:
    now = utc_now_iso()
    with _registry_lock:
        registry = _load_registry_unlocked()
        updated_documents: list[DocumentRecord] = []
        for record in registry.documents:
            if record.document_id in document_ids and record.status == "active":
                updated_documents.append(
                    record.model_copy(update={"last_ingested_at": now})
                )
            else:
                updated_documents.append(record)
        registry.documents = updated_documents
        registry.indexed_content_version = registry.content_version
        _save_registry_unlocked(registry)
