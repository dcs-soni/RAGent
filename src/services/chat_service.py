import hashlib
import json
import logging
from typing import AsyncGenerator

from src.document_store import get_registry_summary

logger = logging.getLogger(__name__)

_graph = None

def load_graph() -> None:
    """Build the graph once at startup instead of per-request."""
    global _graph
    try:
        from src.graph import build_graph
        _graph = build_graph()
        logger.info("✅ Graph compiled at startup")
    except Exception as e:
        logger.warning("⚠️ Graph not available at startup (missing vector DB?): %s", e)
        _graph = None

def unload_graph() -> None:
    global _graph
    _graph = None

def rebuild_chat_graph() -> None:
    global _graph
    load_graph()

def is_graph_ready() -> bool:
    return _graph is not None

def scope_thread_id(api_key: str, thread_id: str) -> str:
    """Namespace thread IDs by API key to prevent cross-tenant state leakage."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return f"{key_hash}:{thread_id}"

async def generate_chat_stream(question: str, scoped_thread_id: str) -> AsyncGenerator[str, None]:
    global _graph
    registry = get_registry_summary()
    
    if registry.active_documents == 0:
        yield f"data: {json.dumps({'error': 'No active documents available. Upload and ingest documents first.'})}\n\n"
        yield "data: [DONE]\n\n"
        return
        
    if registry.needs_reindex:
        yield f"data: {json.dumps({'error': 'Documents changed since the last ingestion. Run ingestion before chatting.'})}\n\n"
        yield "data: [DONE]\n\n"
        return
        
    if _graph is None:
        yield f"data: {json.dumps({'error': 'Agent graph not available. Run document ingestion first.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    initial_state = {
        "question": question,
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
