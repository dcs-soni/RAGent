from __future__ import annotations

import uuid
from threading import Lock

from pydantic import BaseModel

from src.utils import utc_now_iso


class IngestionJob(BaseModel):
    job_id: str
    status: str
    stage: str
    message: str
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    error: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None

    @property
    def progress_percent(self) -> int:
        if self.total_chunks > 0:
            return min(100, int((self.processed_chunks / self.total_chunks) * 100))
        if self.total_documents > 0:
            return min(100, int((self.processed_documents / self.total_documents) * 100))
        return 0

    def to_response(self) -> dict:
        payload = self.model_dump(mode="json")
        payload["progress_percent"] = self.progress_percent
        return payload


class IngestionJobTracker:
    def __init__(self) -> None:
        self._lock = Lock()
        self._jobs: dict[str, IngestionJob] = {}
        self._latest_job_id: str | None = None

    def create_job(self, total_documents: int) -> IngestionJob:
        job = IngestionJob(
            job_id=uuid.uuid4().hex,
            status="queued",
            stage="queued",
            message="Ingestion queued",
            total_documents=total_documents,
            created_at=utc_now_iso(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._latest_job_id = job.job_id
        return job

    def start_job(self, job_id: str, message: str = "Ingestion started") -> IngestionJob:
        return self.update_job(
            job_id,
            status="running",
            stage="starting",
            message=message,
            started_at=utc_now_iso(),
        )

    def update_job(self, job_id: str, **updates) -> IngestionJob:
        with self._lock:
            job = self._jobs[job_id]
            job = job.model_copy(update=updates)
            self._jobs[job_id] = job
            return job

    def finish_job(self, job_id: str, message: str) -> IngestionJob:
        with self._lock:
            job = self._jobs[job_id]
        return self.update_job(
            job_id,
            status="completed",
            stage="completed",
            message=message,
            processed_documents=job.total_documents,
            processed_chunks=job.total_chunks,
            finished_at=utc_now_iso(),
        )

    def fail_job(self, job_id: str, error: str) -> IngestionJob:
        return self.update_job(
            job_id,
            status="failed",
            stage="failed",
            message="Ingestion failed",
            error=error,
            finished_at=utc_now_iso(),
        )

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_latest_job(self) -> IngestionJob | None:
        with self._lock:
            if not self._latest_job_id:
                return None
            return self._jobs.get(self._latest_job_id)


job_tracker = IngestionJobTracker()
