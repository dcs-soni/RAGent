"""
Retrieval & Grading Module
===========================
Handles document retrieval and LLM-based quality grading.

Developer Thinking:
    Retrieval is where most RAG systems silently fail. You pull 4 documents,
    maybe 2 are relevant, and the LLM hallucinates from the irrelevant ones.

    The fix: GRADE every retrieved document before feeding it to the generator.
    This is what makes our system "self-corrective" — it doesn't blindly trust
    retrieval results.

    We implement THREE quality checks:
    1. RELEVANCE GRADING:  Is this document relevant to the question?
    2. HALLUCINATION CHECK: Is the answer grounded in the retrieved documents?
    3. ANSWER QUALITY:     Does the answer actually address the question?

    Each check uses structured output (Pydantic models) so we get reliable
    binary yes/no decisions instead of parsing free-text LLM responses.
"""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Sequence, Optional, Any
from langchain_core.callbacks.manager import Callbacks

from src.config import settings

logger = logging.getLogger(__name__)


# Structured Output Models 
# Using Pydantic models with LLM structured output gives us type-safe,
# parseable responses instead of fragile string parsing.


class RelevanceGrade(BaseModel):
    """Binary relevance grade for a retrieved document."""

    score: str = Field(
        description="Is the document relevant to the question? 'yes' or 'no'."
    )


class HallucinationGrade(BaseModel):
    """Binary hallucination check for a generated answer."""

    score: str = Field(
        description="Is the answer grounded in the provided documents? 'yes' or 'no'."
    )


class AnswerGrade(BaseModel):
    """Binary quality check — does the answer address the question?"""

    score: str = Field(
        description="Does the answer address the question? 'yes' or 'no'."
    )

def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """
    Create a configured LLM instance.

    Args:
        temperature: Controls randomness. 0.0 for grading (deterministic),
                     0.3-0.7 for generation (creative).
    """
    settings.require_google_api_key()
    return ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
    )


class LLMRerankerScore(BaseModel):
    score: int = Field(description="Relevance score from 1 to 10")

class LLMAsJudgeReranker(BaseDocumentCompressor):
    """Uses an LLM to score and rerank documents, acting as a Cross-Encoder."""
    llm: Any = Field(description="The LLM to use for scoring")
    top_n: int = Field(default=4, description="Number of documents to return after reranking")
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
            
        structured_llm = self.llm.with_structured_output(LLMRerankerScore)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Score the relevance of the following document to the user's question on a scale from 1 to 10. Respond with just the integer score."),
            ("human", "Question: {query}\n\nDocument: {document}")
        ])
        chain = prompt | structured_llm
        
        logger.info("⚖️  RERANKING: Scoring %d initial retrieved documents", len(documents))
        scored_docs = []
        for doc in documents:
            try:
                res = chain.invoke({"query": query, "document": doc.page_content})
                scored_docs.append((res.score, doc))
            except Exception as e:
                logger.warning("Reranker failed for a doc: %s", e)
                scored_docs.append((1, doc))
                
        # Sort descending by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:self.top_n]]
        logger.info("  → Reranking complete. Selected top %d documents.", len(top_docs))
        return top_docs

def get_retriever(vector_store: Chroma) -> ContextualCompressionRetriever:
    """
    Create a Hybrid Retriever (Dense + Sparse BM25) paired with an LLM Reranker.
    """
    # 1. Base Dense Retriever (Fetch 2x the final K)
    fetch_k = settings.RETRIEVAL_K * 2
    dense_retriever = vector_store.as_retriever(search_kwargs={"k": fetch_k})
    
    # 2. Base Sparse Retriever (BM25)
    db_data = vector_store.get()
    all_docs = []
    if db_data and "documents" in db_data:
        for i, text in enumerate(db_data["documents"]):
            metadata = db_data["metadatas"][i] if db_data["metadatas"] else {}
            all_docs.append(Document(page_content=text, metadata=metadata))
            
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = fetch_k
        
        base_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, bm25_retriever],
            weights=[0.6, 0.4] # Favor dense slightly
        )
    else:
        base_retriever = dense_retriever
        
    # 3. LLM Cross-Encoder Reranker
    llm = get_llm(temperature=0.0)
    reranker = LLMAsJudgeReranker(llm=llm, top_n=settings.RETRIEVAL_K)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )
    
    return compression_retriever


def grade_document_relevance(document: Document, question: str) -> bool:
    """
    Grade whether a single document is relevant to the question.

    This is the FIRST quality gate. We check each retrieved document
    individually. Irrelevant documents are filtered out before generation.

    Args:
        document: A single retrieved document.
        question: The user's question.

    Returns:
        True if the document is relevant, False otherwise.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(RelevanceGrade)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a document relevance grader. Your job is to determine "
            "whether a retrieved document is relevant to the user's question.\n\n"
            "Rules:\n"
            "- If the document contains keywords or semantic meaning related "
            "to the question, grade it as relevant.\n"
            "- The document does NOT need to fully answer the question — "
            "partial relevance counts.\n"
            "- Respond with a binary 'yes' or 'no' score.",
        ),
        (
            "human",
            "Document content:\n{document}\n\nUser question: {question}",
        ),
    ])

    chain = prompt | structured_llm
    result: RelevanceGrade = chain.invoke({
        "document": document.page_content,
        "question": question,
    })

    is_relevant = result.score.strip().lower() == "yes"
    logger.debug(
        "Relevance grade for chunk from '%s': %s",
        document.metadata.get("source_file", "unknown"),
        "RELEVANT" if is_relevant else "NOT RELEVANT",
    )
    return is_relevant


def check_hallucination(documents: list[Document], generation: str) -> bool:
    """
    Check if the generated answer is grounded in the retrieved documents.

    This is the SECOND quality gate. Even if documents are relevant, the LLM
    might hallucinate facts not present in them.

    Args:
        documents: The filtered, relevant documents.
        generation: The LLM's generated answer.

    Returns:
        True if the answer is grounded (no hallucination), False otherwise.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(HallucinationGrade)

    # Combine all document contents for the check
    combined_docs = "\n\n---\n\n".join(doc.page_content for doc in documents)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a hallucination grader. Your job is to determine whether "
            "an LLM-generated answer is grounded in the provided source documents.\n\n"
            "Rules:\n"
            "- Check if EVERY claim in the answer can be traced back to the documents.\n"
            "- The answer can summarize or paraphrase — it doesn't need exact quotes.\n"
            "- If the answer introduces facts NOT present in the documents, "
            "that's hallucination.\n"
            "- Respond with 'yes' if grounded (no hallucination) or 'no' if "
            "hallucinated.",
        ),
        (
            "human",
            "Source documents:\n{documents}\n\n"
            "Generated answer:\n{generation}",
        ),
    ])

    chain = prompt | structured_llm
    result: HallucinationGrade = chain.invoke({
        "documents": combined_docs,
        "generation": generation,
    })

    is_grounded = result.score.strip().lower() == "yes"
    logger.info("Hallucination check: %s", "GROUNDED" if is_grounded else "HALLUCINATED")
    return is_grounded


def grade_answer_quality(question: str, generation: str) -> bool:
    """
    Check if the generated answer actually addresses the user's question.

    This is the THIRD quality gate. An answer can be grounded in documents
    but still miss the point of the question entirely.

    Args:
        question: The user's original question.
        generation: The LLM's generated answer.

    Returns:
        True if the answer is useful, False otherwise.
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(AnswerGrade)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an answer quality grader. Your job is to determine whether "
            "an answer actually addresses and is useful for the given question.\n\n"
            "Rules:\n"
            "- The answer should directly address what was asked.\n"
            "- Partial answers that provide useful information count as useful.\n"
            "- Generic or evasive answers that don't help are NOT useful.\n"
            "- Respond with 'yes' if useful or 'no' if not useful.",
        ),
        (
            "human",
            "Question: {question}\n\nAnswer: {generation}",
        ),
    ])

    chain = prompt | structured_llm
    result: AnswerGrade = chain.invoke({
        "question": question,
        "generation": generation,
    })

    is_useful = result.score.strip().lower() == "yes"
    logger.info("Answer quality check: %s", "USEFUL" if is_useful else "NOT USEFUL")
    return is_useful
