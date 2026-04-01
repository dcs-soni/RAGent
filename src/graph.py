"""
Graph Builder
==============
Assembles all nodes and edges into a compiled LangGraph StateGraph.

Developer Thinking:
    This is where the architecture comes together. The graph definition
    reads like a flowchart:

        START → retrieve → grade_documents → [decision]
                                               ├─ generate → [quality check]
                                               │               ├─ output (END)
                                               │               ├─ generate (retry)
                                               │               └─ transform_query → retrieve
                                               └─ web_search → generate

    Key design decisions:
    1. CONDITIONAL EDGES make the agent adaptive — it doesn't follow a
       fixed path but reacts to the quality of its own output.
    2. The SELF-CORRECTION LOOP (generate → check → retry) is what separates
       an "agent" from a simple chain.
    3. The MAX_RETRIES escape hatch prevents infinite loops — a production
       MUST-HAVE that tutorials often skip.
    4. WEB SEARCH FALLBACK gives graceful degradation instead of "I don't know."
"""

import logging

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.nodes import generate, grade_documents, retrieve, transform_query, web_search
from src.edges import check_generation_quality, decide_to_generate
from src.state import GraphState

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """
    Build and compile the Self-Corrective RAG workflow graph.

    Graph topology:

        ┌──────────┐
        │  START    │
        └────┬─────┘
             ▼
        ┌──────────┐
        │ retrieve  │
        └────┬─────┘
             ▼
        ┌────────────────┐
        │ grade_documents │
        └────┬───────────┘
             ▼
        ┌─────────────────────┐
        │ decide_to_generate  │ (conditional edge)
        ├─────────┬───────────┤
        │relevant │ no docs   │
        ▼         ▼           │
    ┌──────┐  ┌───────────┐   │
    │generate│  │web_search │  │
    └──┬───┘  └─────┬─────┘   │
       │            │          │
       │      ┌─────▼─────┐   │
       │      │  generate  │◄──┘
       │      └─────┬──────┘
       ▼            ▼
    ┌────────────────────────┐
    │ check_generation_quality│ (conditional edge)
    ├────────┬────────┬──────┤
    │output  │generate│transform
    ▼        ▼        ▼
    END    (retry)  ┌──────────────┐
                    │transform_query│
                    └──────┬───────┘
                           ▼
                    ┌──────────┐
                    │ retrieve  │ (loop back)
                    └──────────┘

    Returns:
        A compiled LangGraph ready for invocation.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)


    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("web_search", "generate")

    workflow.add_conditional_edges(
        "generate",
        check_generation_quality,
        {
            "output": END,
            "generate": "generate",         # Retry generation
            "transform_query": "transform_query",  # Rewrite query
        },
    )

    workflow.add_edge("transform_query", "retrieve")

    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)
    logger.info("✅ Graph compiled successfully with MemorySaver Checkpointer")
    return compiled
