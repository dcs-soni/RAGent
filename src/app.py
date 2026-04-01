"""
Streamlit Chat Interface
=========================
Production-quality chat UI with session state, graph execution tracing,
and document upload.

Developer Thinking:
    The UI is NOT an afterthought — it's how users experience the system.
    Key production considerations:
    1. SESSION STATE: Streamlit reruns the entire script on every interaction.
       We must persist chat history and the graph instance in st.session_state.
    2. EXECUTION TRACE: Show users WHICH nodes ran and WHY decisions were made.
       This builds trust and makes debugging transparent.
    3. ERROR HANDLING: Never show raw Python tracebacks. Catch errors and show
       friendly messages with actionable next steps.
    4. UPLOAD FLOW: Let users upload PDFs directly from the UI, trigger
       ingestion, and start chatting — zero CLI required.
"""
import uuid
import logging
import os
import requests
import json

import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "30"))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Self-Corrective RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown("""
<style>
    /* Chat container */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Sidebar sections */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Trace log */
    .trace-entry {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.75rem;
        padding: 0.25rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Status indicators */
    .status-ready { color: #4CAF50; }
    .status-no-docs { color: #FF9800; }
    .status-error { color: #F44336; }
</style>
""", unsafe_allow_html=True)




def init_session_state():
    """Initialize all session state variables on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "trace_log" not in st.session_state:
        st.session_state.trace_log = []
    if "docs_ingested" not in st.session_state:
        st.session_state.docs_ingested = False
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())


init_session_state()


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    return payload.get("detail") or payload.get("message") or response.text or f"HTTP {response.status_code}"


def api_request(method: str, path: str, **kwargs):
    response = requests.request(
        method,
        f"{BACKEND_URL}{path}",
        timeout=kwargs.pop("timeout", HTTP_TIMEOUT_SECONDS),
        **kwargs,
    )
    if not response.ok:
        raise RuntimeError(_extract_error_message(response))
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    return response.text


def fetch_sidebar_state() -> tuple[dict, dict, dict]:
    documents_payload = {"documents": [], "summary": {}}
    ingestion_payload = {
        "status": "idle",
        "stage": "idle",
        "message": "No ingestion job has been started yet",
        "progress_percent": 0,
    }
    health_payload = {"status": "offline"}

    try:
        documents_payload = api_request("GET", "/documents")
    except Exception:
        logger.exception("Failed to fetch document registry")

    try:
        ingestion_payload = api_request("GET", "/ingest/status")
    except Exception:
        logger.exception("Failed to fetch ingestion status")

    try:
        health_payload = api_request("GET", "/health")
    except Exception:
        logger.exception("Failed to fetch backend health")

    return documents_payload, ingestion_payload, health_payload

def render_sidebar():
    """Render the sidebar with document upload, status, and execution trace."""
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        documents_payload, ingestion_payload, health_payload = fetch_sidebar_state()
        documents = documents_payload.get("documents", [])
        summary = documents_payload.get("summary", {})

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to build the knowledge base.",
        )

        if uploaded_files and st.button("⬆️ Upload Selected Files", use_container_width=True):
            try:
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    api_request("POST", "/documents", files=files)
                st.session_state.docs_ingested = False
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded to backend")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

        if documents:
            st.markdown("### Managed Documents")
            for document in documents:
                title = f"{document['original_filename']}  v{document['version']}"
                with st.expander(title):
                    st.caption(f"ID: `{document['document_id']}`")
                    st.caption(f"Size: {document['size_bytes']} bytes")
                    st.caption(f"Updated: {document['updated_at']}")
                    if document.get("last_ingested_at"):
                        st.caption(f"Indexed: {document['last_ingested_at']}")
                    if st.button("Delete", key=f"delete_{document['document_id']}", use_container_width=True):
                        try:
                            api_request("DELETE", f"/documents/{document['document_id']}")
                            st.session_state.docs_ingested = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Delete failed: {e}")

            st.markdown("### Replace Document")
            replace_options = {
                f"{document['original_filename']} ({document['document_id'][:8]})": document["document_id"]
                for document in documents
            }
            selected_label = st.selectbox(
                "Choose a document to replace",
                options=list(replace_options.keys()),
            )
            replacement_file = st.file_uploader(
                "Upload replacement PDF",
                type=["pdf"],
                accept_multiple_files=False,
                key="replacement_file",
            )
            if replacement_file and st.button("♻️ Replace Selected Document", use_container_width=True):
                try:
                    files = {"file": (replacement_file.name, replacement_file.getvalue(), "application/pdf")}
                    api_request(
                        "PUT",
                        f"/documents/{replace_options[selected_label]}",
                        files=files,
                    )
                    st.session_state.docs_ingested = False
                    st.success("✅ Document replaced")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Update failed: {e}")

        if st.button("🔄 Ingest Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing documents in background..."):
                try:
                    response = api_request("POST", "/ingest")
                    st.session_state.docs_ingested = True
                    st.success(
                        f"✅ Ingestion started: job `{response['job']['job_id'][:8]}`"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

        st.markdown("---")
        st.markdown("## 🧭 Ingestion Status")
        st.caption(ingestion_payload.get("message", "No ingestion activity"))
        st.progress(int(ingestion_payload.get("progress_percent", 0)))
        st.caption(
            f"Stage: {ingestion_payload.get('stage', 'idle')} | "
            f"Status: {ingestion_payload.get('status', 'idle')}"
        )
        total_documents = ingestion_payload.get("total_documents", 0)
        processed_documents = ingestion_payload.get("processed_documents", 0)
        total_chunks = ingestion_payload.get("total_chunks", 0)
        processed_chunks = ingestion_payload.get("processed_chunks", 0)
        if total_documents:
            st.caption(f"Documents: {processed_documents}/{total_documents}")
        if total_chunks:
            st.caption(f"Chunks: {processed_chunks}/{total_chunks}")
        if ingestion_payload.get("error"):
            st.error(ingestion_payload["error"])
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()

        st.markdown("---")
        st.markdown("## 📊 System Status")

        if health_payload.get("status") == "ok":
            st.markdown("🟢 **Backend Engine:** Ready")
        elif health_payload.get("status") == "stale_index":
            st.markdown("🟠 **Backend Engine:** Re-ingestion required")
        elif health_payload.get("status") == "no_documents":
            st.markdown("🟡 **Backend Engine:** No documents uploaded")
        elif health_payload.get("status") == "no_vector_db":
            st.markdown("🟡 **Backend Engine:** Documents uploaded, ingestion needed")
        else:
            st.markdown("🔴 **Backend Engine:** Offline")
        if summary:
            st.caption(
                f"Active docs: {summary.get('active_documents', 0)} | "
                f"Needs reindex: {summary.get('needs_reindex', False)}"
            )

        if st.session_state.trace_log:
            st.markdown("---")
            st.markdown("## 🔍 Execution Trace")
            st.markdown(
                "Shows which graph nodes ran for the last query."
            )
            for entry in st.session_state.trace_log:
                st.markdown(f"`{entry}`")

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.trace_log = []
            st.rerun()


def stream_graph_response(question: str):
    """
    Connect to FastAPI SSE endpoint and parse state updates in real-time.
    Yields intermediate trace steps and returns the final answer.
    """
    st.session_state.trace_log = []
    answer = "Error: no answer generated"
    
    try:
        payload = {
            "question": question,
            "thread_id": st.session_state.thread_id
        }
        with requests.post(
            f"{BACKEND_URL}/chat",
            json=payload,
            stream=True,
            timeout=(5, HTTP_TIMEOUT_SECONDS),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith("data: "):
                        data = decoded[len("data: "):]
                        if data == "[DONE]":
                            break
                        
                        event_dict = json.loads(data)
                        if "error" in event_dict:
                            return f"❌ **Error:** {event_dict['error']}"
                            
                        # Update trace log
                        node = event_dict.get("node")
                        st.session_state.trace_log.append(f"▶ {node}")
                        
                        # Show trace updates instantly in the sidebar via st.sidebar? 
                        # Streamlit reruns make dynamically updating the sidebar hard.
                        # We'll just collect them and they show on next rerun.
                        
                        if event_dict.get("generation"):
                            answer = event_dict.get("generation")
                            
        return answer

    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error:** Could not connect to the Backend API. Make sure it is running on port 8000."
    except Exception as e:
        logger.exception("Graph execution failed")
        return f"❌ **Error during processing:** {e}"

def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()

    st.markdown(
        """
        # 🤖 Self-Corrective RAG Agent

        Ask questions about your uploaded documents. The agent retrieves relevant
        information, verifies its own answers, and self-corrects when needed.
        """
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Connecting to Agent API..."):
                # We can't easily stream the chunks to `write_stream` without a generator,
                # but we are displaying the trace changes.
                response = stream_graph_response(prompt)
                st.markdown(response)

        # Add assistant response to history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


if __name__ == "__main__":
    main()
