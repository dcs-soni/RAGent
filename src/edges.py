"""
Conditional Edges
==================
Edge functions decide WHERE to route the graph after a node completes.

Developer Thinking:
    In a simple chain, data flows A → B → C linearly. In a LangGraph,
    we add CONDITIONAL edges that route to different nodes based on the
    current state. This is what gives the agent its "intelligence":

    - After grading, decide: generate OR web search?
    - After generating, decide: output OR retry OR rewrite?

    These are simple if/else functions, but they encode the agent's
    decision-making logic. They're also the easiest place to add
    business rules ("never retry more than 3 times").
"""

import logging

from src.config import settings
from src.retrieval import check_hallucination, grade_answer_quality
from src.state import GraphState

logger = logging.getLogger(__name__)


def decide_to_generate(state: GraphState) -> str:
    """
    Route after document grading: generate or web search?

    Decision logic:
    - If web_search_needed is True (no relevant docs found) → web search
    - Otherwise → generate an answer from the relevant docs

    Returns:
        The name of the next node to execute.
    """
    if state.get("web_search_needed", False):
        logger.info("🔀 ROUTING → web_search (no relevant documents)")
        return "web_search"

    logger.info("🔀 ROUTING → generate (relevant documents found)")
    return "generate"


def check_generation_quality(state: GraphState) -> str:
    """
    Route after generation: accept, retry, or rewrite?

    This is the SELF-CORRECTION loop. After generating an answer, we check:
    1. Is the answer grounded in the documents? (hallucination check)
    2. Does the answer actually address the question? (quality check)

    Decision tree:
    ┌─────────────────────────────┐
    │ retry_count >= MAX_RETRIES? │
    │  YES → output (give up)    │
    │  NO  ↓                     │
    ├─────────────────────────────┤
    │ Is answer grounded?        │
    │  NO  → generate (retry)    │
    │  YES ↓                     │
    ├─────────────────────────────┤
    │ Does answer address Q?     │
    │  NO  → transform_query     │
    │  YES → output (success!)   │
    └─────────────────────────────┘

    Returns:
        The name of the next node: "output", "generate", or "transform_query".
    """
    retry_count = state.get("retry_count", 0)
    generation = state.get("generation", "")
    question = state["question"]
    documents = state["documents"]

    # Safety valve: prevent infinite loops
    if retry_count >= settings.MAX_RETRIES:
        logger.warning(
            "⚠️ Max retries (%d) reached — outputting best effort answer",
            settings.MAX_RETRIES,
        )
        return "output"

    logger.info("🔍 Checking hallucination (attempt %d/%d)", retry_count, settings.MAX_RETRIES)
    is_grounded = check_hallucination(documents, generation)

    if not is_grounded:
        logger.info("🔀 ROUTING → generate (hallucination detected, retrying)")
        return "generate"

    logger.info("🔍 Checking answer quality")
    is_useful = grade_answer_quality(question, generation)

    if not is_useful:
        logger.info("🔀 ROUTING → transform_query (answer not useful, rewriting query)")
        return "transform_query"

    logger.info("✅ ROUTING → output (grounded + useful)")
    return "output"
