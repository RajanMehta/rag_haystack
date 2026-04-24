import json
import logging
from typing import Dict, List, Union

from fastapi import APIRouter
from pydantic import BaseModel, Field

from haystack_api.config import LOG_LEVEL
from haystack_api.utils import get_app

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")

router = APIRouter()
app = get_app()


class IREvaluationRequest(BaseModel):
    """
    Request model for information retrieval evaluation endpoint.
    """

    ground_truth_documents: Union[List[str], str] = Field(
        description="List of relevant document IDs for the query. Can be provided as a list or a string representation "
        "of a list."
    )
    retrieved_documents: Union[List[Dict[str, Union[str, float]]], str] = Field(
        description="List of retrieved documents with their scores. Each document should have 'id' and 'score' fields. "
        "Can be provided as a list or a string representation of a list."
    )
    k: int = Field(default=10, description="The k value to use for evaluation metrics.")


class IRMetricsResponse(BaseModel):
    """
    Response model for information retrieval evaluation endpoint.
    """

    accuracy: float = Field(description="Accuracy at specified k value.")
    precision: float = Field(description="Precision at specified k value.")
    recall: float = Field(description="Recall at specified k value.")
    mrr: float = Field(description="Mean Reciprocal Rank.")
    map: float = Field(description="Mean Average Precision.")


def compute_metrics(
    retrieved_docs: List[Dict[str, Union[str, float]]], ground_truth_docs: List[str], k: int
) -> Dict[str, float]:
    """
    Compute IR metrics directly from retrieved documents and ground truth.

    Args:
        retrieved_docs: List of retrieved documents with their scores
        ground_truth_docs: List of relevant document IDs
        k: The k value to use for evaluation metrics

    Returns:
        Dictionary containing the computed metrics
    """
    if not ground_truth_docs:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "map": 0.0,
        }

    # 1. Handle Duplicates: Ensure unique document IDs, keeping the highest score
    unique_docs = {}
    for doc in retrieved_docs:
        doc_id = str(doc["id"])
        doc_score = float(doc["score"])
        if doc_id not in unique_docs or doc_score > unique_docs[doc_id]["score"]:
            unique_docs[doc_id] = {"id": doc_id, "score": doc_score}

    # Sort retrieved documents by score and limit to top k
    sorted_docs = sorted(unique_docs.values(), key=lambda x: x["score"], reverse=True)
    top_k_hits = sorted_docs[:k]

    ground_truth_set = set(ground_truth_docs)

    # Accuracy@k: 1 if at least one relevant document is in top-k, 0 otherwise
    accuracy = 0.0
    for hit in top_k_hits:
        if hit["id"] in ground_truth_set:
            accuracy = 1.0
            break

    # Precision@k & Recall@k
    num_correct_in_top_k = sum(1 for hit in top_k_hits if hit["id"] in ground_truth_set)

    precision = num_correct_in_top_k / k if k > 0 else 0.0
    recall = num_correct_in_top_k / len(ground_truth_set)

    # MRR@k: Mean Reciprocal Rank
    mrr = 0.0
    for rank, hit in enumerate(top_k_hits):
        if hit["id"] in ground_truth_set:
            mrr = 1.0 / (rank + 1)
            break

    # MAP@k: Mean Average Precision
    map_sum = 0.0
    num_correct_so_far = 0
    for rank, hit in enumerate(top_k_hits):
        if hit["id"] in ground_truth_set:
            num_correct_so_far += 1
            precision_at_rank = num_correct_so_far / (rank + 1)
            map_sum += precision_at_rank

    # Following sentence-transformers' logic for the denominator
    denominator = min(k, len(ground_truth_set))
    map_score = map_sum / denominator if denominator > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "map": map_score,
    }


@router.post(
    "/evaluators/information-retrieval",
    response_model=IRMetricsResponse,
    summary="Evaluate information retrieval metrics",
)
async def evaluate_information_retrieval(request: IREvaluationRequest) -> Dict:
    """
    Evaluates information retrieval performance using standard IR metrics.
    Accepts retrieved documents and ground truth document IDs as input.

    Example request:
    ```json
    {
        "ground_truth_documents": ["doc1", "doc2"],
        "retrieved_documents": [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.8},
            {"id": "doc3", "score": 0.7}
        ],
        "k": 5
    }
    ```
    """
    ground_truth_docs = request.ground_truth_documents
    retrieved_docs = request.retrieved_documents
    k_value = request.k

    # Convert string to list if necessary
    if isinstance(ground_truth_docs, str):
        safe_json_str = ground_truth_docs.replace("'", '"')
        ground_truth_docs = json.loads(safe_json_str)
    if isinstance(retrieved_docs, str):
        retrieved_docs = json.loads(retrieved_docs)

    # Compute metrics
    metrics = compute_metrics(retrieved_docs, ground_truth_docs, k_value)

    return metrics
