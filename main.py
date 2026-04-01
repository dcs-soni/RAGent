"""
Main Entry Point
==================
CLI interface for running ingestion or launching the Streamlit app.

Usage:
    python main.py ingest          # Ingest PDFs from docs/ folder
    python main.py ingest /path    # Ingest PDFs from a custom folder
    streamlit run src/app.py       # Launch the chat UI
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_usage():
    """Print CLI usage instructions."""
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║           Self-Corrective RAG Agent with LangGraph          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Commands:                                                   ║
║    python main.py ingest [path]   Ingest PDF documents       ║
║    streamlit run src/app.py       Launch the chat UI         ║
║                                                              ║
║  Quick Start:                                                ║
║    1. Copy .env.example to .env and add your API key         ║
║    2. Put PDF files in the docs/ folder                      ║
║    3. Run: python main.py ingest                             ║
║    4. Run: streamlit run src/app.py                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
    )


def cmd_ingest(docs_dir: str | None = None):
    """Run the document ingestion pipeline."""
    from src.ingestion import ingest_pipeline

    try:
        vector_store = ingest_pipeline(docs_dir)
        logger.info("Ingestion complete! You can now run: streamlit run src/app.py")
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception:
        logger.exception("Ingestion failed unexpectedly")
        sys.exit(1)


def main():
    """Parse CLI arguments and dispatch to the appropriate command."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()

    if command == "ingest":
        docs_dir = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_ingest(docs_dir)
    elif command == "help":
        print_usage()
    else:
        print(f"Unknown command: '{command}'")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
