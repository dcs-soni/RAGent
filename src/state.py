"""
Graph State Definition
======================
Defines the shared state that flows through every node in the LangGraph workflow.

Developer Thinking:
    LangGraph works by passing a "state" dictionary through a graph of nodes.
    Every node reads from and writes to this state. Using TypedDict gives us:
    - Clear documentation of what data flows through the system
    - IDE type hints for every field
    - Easy debugging (just print the state at any node)

    We use plain list[Document] (not Annotated with operator.add) because
    the grade_documents node needs to REPLACE the full document list with
    only the relevant documents, not append to it.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Shared state for the Self-Corrective RAG workflow.

    Attributes:
        question:          The user's original question.
        documents:         Retrieved / filtered documents.
        generation:        The LLM-generated answer.
        web_search_needed: Flag indicating retrieval quality was insufficient.
        retry_count:       Number of self-correction attempts so far.
    """

    question: str
    documents: list[Document]
    generation: str
    web_search_needed: bool
    retry_count: int
