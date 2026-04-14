import logging
from threading import Lock

from fastapi import BackgroundTasks
from src.document_store import get_registry_summary
from src.ingestion_jobs import job_tracker

logger = logging.getLogger(__name__)

ingestion_lock = Lock()
ingestion_running = False

def is_ingestion_running() -> bool:
    return ingestion_running

def assert_mutations_allowed() -> None:
    if is_ingestion_running():
        raise ValueError("Document changes are disabled while ingestion is running")

def get_latest_job_response() -> dict:
    latest_job = job_tracker.get_latest_job()
    if latest_job is None:
        return {
            "status": "idle",
            "stage": "idle",
            "message": "No ingestion job has been started yet",
            "progress_percent": 0,
        }
    return latest_job.to_response()

def trigger_background_ingestion(background_tasks: BackgroundTasks):
    global ingestion_running
    registry = get_registry_summary()
    
    if registry.active_documents == 0:
        raise ValueError("No active documents found. Upload documents first.")

    with ingestion_lock:
        if ingestion_running:
            raise RuntimeError("Ingestion is already running")
        ingestion_running = True

    try:
        job = job_tracker.create_job(total_documents=registry.active_documents)
    except Exception:
        with ingestion_lock:
            ingestion_running = False
        raise RuntimeError("Failed to create ingestion job")

    def _ingest_and_rebuild():
        global ingestion_running
        try:
            from src.services.chat_service import rebuild_chat_graph
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

            from src.ingestion import ingest_pipeline
            ingest_pipeline(progress_callback=_progress_update)
            
            # Rebuild graph
            rebuild_chat_graph()
            
            job_tracker.finish_job(job.job_id, "Ingestion completed successfully")
            logger.info("✅ Graph rebuilt after ingestion")
        except Exception as exc:
            job_tracker.fail_job(job.job_id, str(exc))
            logger.exception("Background ingestion failed")
        finally:
            with ingestion_lock:
                ingestion_running = False

    background_tasks.add_task(_ingest_and_rebuild)
    return job_tracker.get_job(job.job_id).to_response()
