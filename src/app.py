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

def render_sidebar():
    """Render the sidebar with document upload, status, and execution trace."""
    with st.sidebar:
        st.markdown("## 📁 Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to build the knowledge base.",
        )

        if uploaded_files:
            try:
                for uploaded_file in uploaded_files:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{BACKEND_URL}/upload", files=files)
                    res.raise_for_status()
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded to backend")
            except Exception as e:
                st.error(f"❌ Upload failed: {e}")

        if st.button("🔄 Ingest Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing documents in background..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/ingest")
                    res.raise_for_status()
                    st.session_state.docs_ingested = True
                    st.success("✅ Documents ingestion started on the backend!")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")

        st.markdown("---")
        st.markdown("## 📊 System Status")

        try:
            health = requests.get(f"{BACKEND_URL}/health").json()
            if health.get("status") == "ok":
                st.markdown("🟢 **Backend Engine:** Ready")
            else:
                st.markdown("🟡 **Backend Engine:** No vectors found")
        except Exception:
            st.markdown("🔴 **Backend Engine:** Offline")

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
        with requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True) as response:
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
