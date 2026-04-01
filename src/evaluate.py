"""
Production Evaluation Pipeline
==============================
A RAGAS-based evaluation script to continuously measure answer quality,
hallucination, and retrieval precision.
"""

import logging
import json
import uuid
from pathlib import Path

# Ragas requires these to be installed
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevance, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError:
    raise ImportError("Please install ragas and datasets: pip install ragas datasets")

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import settings
from src.graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_evaluation(
    questions_file: str = "test_dataset.json",
    output_file: str = "eval_results.csv",
):
    """
    Run RAGAS evaluation on a test dataset.
    The dataset should be a JSON array of dicts with 'question' and 'ground_truth'.
    """
    filepath = Path(questions_file)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Test dataset '{questions_file}' not found. "
            "Create a JSON array with 'question' and optional 'ground_truth' fields."
        )

    with open(filepath, "r") as f:
        qa_pairs = json.load(f)

    logger.info("Loaded %d questions for evaluation.", len(qa_pairs))

    graph = build_graph()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in qa_pairs:
        q = item["question"]
        gt = item.get("ground_truth", "")

        logger.info("Evaluating Q: %s", q)

        state = {
            "question": q,
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "retry_count": 0,
        }

        # Invoke with config — MemorySaver checkpointer requires thread_id
        try:
            config = {"configurable": {"thread_id": f"eval_{uuid.uuid4().hex[:8]}"}}
            res = graph.invoke(state, config=config)
            ans = res.get("generation", "No generation")
            docs = res.get("documents", [])
            doc_texts = [doc.page_content for doc in docs]
        except Exception as e:
            logger.error("Error during graph execution: %s", e)
            ans = "Error"
            doc_texts = ["Error"]

        questions.append(q)
        answers.append(ans)
        ground_truths.append(gt)
        contexts.append(doc_texts)

    # Prepare dataset for RAGAS
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Initialize RAGAS evaluator models using Google GenAI APIs
    settings.require_google_api_key()
    ragas_llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL, google_api_key=settings.GOOGLE_API_KEY
    )
    ragas_embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY
    )

    logger.info("Analyzing with RAGAS metrics: Faithfulness, Answer Relevance, Context Precision...")

    eval_llm = LangchainLLMWrapper(ragas_llm)
    eval_embeddings = LangchainEmbeddingsWrapper(ragas_embeddings)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevance, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    logger.info("✅ Evaluation Complete!")
    logger.info("================ Results ================")
    logger.info("%s", result)

    # Save results
    try:
        res_df = result.to_pandas()
        res_df.to_csv(output_file, index=False)
        logger.info("Detailed results saved to %s", output_file)
    except Exception as e:
        logger.warning("Could not save to CSV: %s", e)

    return result


if __name__ == "__main__":
    run_evaluation()
