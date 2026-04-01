"""
Document Ingestion Pipeline
============================
Loads PDFs, splits them into chunks, and stores them in a ChromaDB vector store.

Developer Thinking:
    The ingestion pipeline is the FOUNDATION of any RAG system. If your chunks
    are bad, your retrieval is bad, and your answers are bad. No amount of
    fancy prompting can fix garbage retrieval.

    Key decisions:
    1. CHUNK SIZE (~1000 chars): Small enough for precise retrieval, large
       enough to preserve context. Too small → fragments lack meaning.
       Too large → retrieval pulls in irrelevant noise.

    2. CHUNK OVERLAP (~200 chars): Prevents information loss at chunk
       boundaries. A sentence split across two chunks would be lost without
       overlap.

    3. RECURSIVE SPLITTING: We use RecursiveCharacterTextSplitter because it
       tries to respect natural boundaries (paragraphs → sentences → words)
       rather than splitting mid-sentence.

    4. PERSISTENCE: ChromaDB persists to disk so we don't re-embed documents
       every time the app starts. Embedding is the slowest and most expensive
       step.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.document_store import get_active_document_paths, mark_indexed

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict], None]


def _report_progress(progress_callback: ProgressCallback | None, **payload: object) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def load_documents(
    docs_dir: str | Path | None = None,
) -> tuple[list[Document], list[str], int]:
    """
    Load all PDFs from the given directory.

    Args:
        docs_dir: Path to folder containing PDFs. Defaults to settings.DOCS_DIR.

    Returns:
        Tuple of page-level LangChain documents, stable document IDs, and the
        total number of source files loaded.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
        ValueError: If no PDFs are found.
    """
    if docs_dir:
        docs_path = Path(docs_dir).resolve()
        if not docs_path.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {docs_path}. "
                f"Create it and add your PDF files."
            )
        source_entries = [
            {
                "document_id": pdf_file.stem,
                "display_name": pdf_file.name,
                "path": pdf_file,
            }
            for pdf_file in sorted(docs_path.glob("*.pdf"))
        ]
    else:
        source_entries = [
            {
                "document_id": record.document_id,
                "display_name": record.original_filename,
                "path": file_path,
            }
            for record, file_path in get_active_document_paths()
        ]

    if not source_entries:
        raise ValueError(
            "No active PDF documents found. Upload documents before running ingestion."
        )

    all_documents: list[Document] = []
    document_ids: list[str] = []

    for entry in source_entries:
        pdf_file = entry["path"]
        display_name = entry["display_name"]
        document_id = entry["document_id"]
        logger.info("Loading: %s", display_name)
        try:
            loader = PyPDFLoader(str(pdf_file))
            pages = loader.load()

            # Enrich metadata so we can trace answers back to source files
            for page in pages:
                page.metadata["source_file"] = display_name
                page.metadata["document_id"] = document_id

            all_documents.extend(pages)
            logger.info("  → %d pages loaded", len(pages))
            document_ids.append(document_id)
        except Exception:
            logger.exception("Failed to load %s, skipping", display_name)

    if not all_documents:
        raise ValueError("No readable PDF content found. Check that the documents are valid and not encrypted.")

    logger.info(
        "Total documents loaded: %d pages from %d files",
        len(all_documents),
        len(document_ids),
    )
    return all_documents, document_ids, len(source_entries)




def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks using recursive character splitting.

    This respects natural text boundaries (paragraphs, sentences) and adds
    overlap to prevent information loss at chunk edges.

    Args:
        documents: Raw documents from the loader.

    Returns:
        List of chunked documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try paragraphs first, then sentences
        add_start_index=True,  # Track where each chunk came from in the original doc
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d documents → %d chunks (size=%d, overlap=%d)",
        len(documents),
        len(chunks),
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
    )
    return chunks



def apply_contextual_chunking(
    chunks: list[Document],
    all_documents: list[Document],
    progress_callback: ProgressCallback | None = None,
    total_documents: int = 0,
) -> list[Document]:
    """
    Applies Contextual Chunking by asking the LLM to situate each chunk
    within the overall context of its source document.
    """
    logger.info("Applying Contextual Chunking via LLM... (This takes a moment)")
    settings.require_google_api_key()
    
    # Truncate document text to stay within LLM context window.
    # Gemini Flash supports ~1M tokens, but sending full books is slow and costly.
    # 50K chars ≈ ~12K tokens — enough context to situate any chunk.
    MAX_DOC_CHARS = 50_000
    
    # 1. Reconstruct full text per source file
    full_texts = {}
    for doc in all_documents:
        source = doc.metadata.get("source_file", "unknown")
        full_texts[source] = full_texts.get(source, "") + "\n" + doc.page_content
        
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    prompt = ChatPromptTemplate.from_template(
        "<document>\n{document}\n</document>\n\n"
        "Here is the chunk we want to situate within the whole document:\n"
        "<chunk>\n{chunk}\n</chunk>\n\n"
        "Please give a short succinct context to situate this chunk within the overall document "
        "for the purposes of improving search retrieval of the chunk. "
        "Answer only with the succinct context and nothing else."
    )
    
    chain = prompt | llm
    
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source_file", "unknown")
        doc_text = full_texts.get(source, "")
        
        if not doc_text:
            enriched_chunks.append(chunk)
            continue
            
        try:
            # Truncate to prevent exceeding LLM context window
            truncated_doc = doc_text[:MAX_DOC_CHARS]
            if len(doc_text) > MAX_DOC_CHARS:
                logger.debug("  Truncated doc '%s' from %d to %d chars", source, len(doc_text), MAX_DOC_CHARS)
            res = chain.invoke({"document": truncated_doc, "chunk": chunk.page_content})
            context = res.content.strip()
            new_content = f"Context: {context}\n\nChunk: {chunk.page_content}"
            enriched_chunks.append(Document(page_content=new_content, metadata=chunk.metadata.copy()))
        except Exception as e:
            logger.warning("Failed to contextualize chunk %d: %s", i, e)
            enriched_chunks.append(chunk)

        _report_progress(
            progress_callback,
            stage="contextualizing",
            message=f"Contextualized chunk {i + 1} of {len(chunks)}",
            total_documents=total_documents,
            processed_documents=total_documents if total_documents else 0,
            total_chunks=len(chunks),
            processed_chunks=i + 1,
        )
        time.sleep(0.5)  # Throttle to respect free-tier API rate limits
        
    logger.info("Contextual Chunking Complete: Enriched %d chunks", len(enriched_chunks))
    return enriched_chunks


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the embedding model instance."""
    settings.require_google_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )


def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Create a ChromaDB vector store from document chunks.

    The index is rebuilt into a temporary directory and then swapped into
    place, so repeated ingestion does not accumulate duplicate vectors.

    Args:
        chunks: Pre-split document chunks.

    Returns:
        A persisted Chroma vector store.
    """
    embeddings = get_embeddings()

    target_path = settings.chroma_path
    temp_path = target_path.parent / f"{target_path.name}.tmp"
    backup_path = target_path.parent / f"{target_path.name}.bak"

    logger.info("Creating vector store at: %s", target_path)
    shutil.rmtree(temp_path, ignore_errors=True)
    shutil.rmtree(backup_path, ignore_errors=True)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(temp_path),
        collection_name="rag_documents",
    )

    if target_path.exists():
        target_path.replace(backup_path)
    temp_path.replace(target_path)
    shutil.rmtree(backup_path, ignore_errors=True)

    vector_store = Chroma(
        persist_directory=str(target_path),
        embedding_function=embeddings,
        collection_name="rag_documents",
    )
    logger.info("Vector store created with %d chunks", len(chunks))
    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Returns:
        The persisted Chroma vector store.

    Raises:
        FileNotFoundError: If no vector store exists yet.
    """
    if not settings.chroma_path.exists():
        raise FileNotFoundError(
            f"No vector store found at {settings.chroma_path}. "
            f"Run 'python main.py ingest' first to create it."
        )

    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=str(settings.chroma_path),
        embedding_function=embeddings,
        collection_name="rag_documents",
    )

    try:
        count = len(vector_store.get()["ids"])
    except Exception:
        count = -1  # Unknown count — ChromaDB API may have changed
    logger.info("Loaded vector store with %d chunks", count)
    return vector_store




def ingest_pipeline(
    docs_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Chroma:
    """
    Run the full ingestion pipeline: Load → Split → Embed → Store.

    Args:
        docs_dir: Path to PDFs directory. Defaults to settings.DOCS_DIR.

    Returns:
        The populated vector store, ready for retrieval.
    """
    logger.info("=" * 60)
    logger.info("STARTING INGESTION PIPELINE")
    logger.info("=" * 60)

    _report_progress(
        progress_callback,
        stage="loading",
        message="Loading documents",
        total_documents=0,
        processed_documents=0,
        total_chunks=0,
        processed_chunks=0,
    )
    documents, document_ids, total_documents = load_documents(docs_dir)
    _report_progress(
        progress_callback,
        stage="loaded",
        message=f"Loaded {total_documents} documents",
        total_documents=total_documents,
        processed_documents=total_documents,
        total_chunks=0,
        processed_chunks=0,
    )

    _report_progress(
        progress_callback,
        stage="splitting",
        message="Splitting documents into chunks",
        total_documents=total_documents,
        processed_documents=total_documents,
        total_chunks=0,
        processed_chunks=0,
    )
    raw_chunks = split_documents(documents)
    _report_progress(
        progress_callback,
        stage="split_complete",
        message=f"Created {len(raw_chunks)} raw chunks",
        total_documents=total_documents,
        processed_documents=total_documents,
        total_chunks=len(raw_chunks),
        processed_chunks=0,
    )

    enriched_chunks = apply_contextual_chunking(
        raw_chunks,
        documents,
        progress_callback=progress_callback,
        total_documents=total_documents,
    )
    _report_progress(
        progress_callback,
        stage="indexing",
        message="Writing vectors to ChromaDB",
        total_documents=total_documents,
        processed_documents=total_documents,
        total_chunks=len(enriched_chunks),
        processed_chunks=len(enriched_chunks),
    )
    vector_store = create_vector_store(enriched_chunks)

    if docs_dir is None:
        mark_indexed(document_ids)

    logger.info("INGESTION COMPLETE — %d chunks indexed", len(enriched_chunks))
    logger.info("=" * 60)
    _report_progress(
        progress_callback,
        stage="completed",
        message=f"Indexed {len(enriched_chunks)} chunks from {total_documents} documents",
        total_documents=total_documents,
        processed_documents=total_documents,
        total_chunks=len(enriched_chunks),
        processed_chunks=len(enriched_chunks),
    )
    return vector_store
