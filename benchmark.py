"""Benchmark scores for the embeddings of a model.

The unit score of a query-image embedding is whether, among the instruction images ranked by
similarity (same model, same build), one of the top-k has the same step number as the query.
"""

import warnings

from data.images_store import ImageRecord
from data.query_processor import load_model_embeddings
import faiss
import numpy as np
import torch

TOP_K: list[int] = [1, 3, 5]

def _compute_top_k_accuracies(
    image_record: ImageRecord,
    image_embedding: torch.Tensor,
    index: tuple[faiss.Index, list[ImageRecord]],
) -> dict[str, float]:
    """Return {k: 1.0 if one of the k nearest instruction images has the query's step number else 0.0}."""
    faiss_index, records = index
    #the index holds L2-normalized vectors (cosine)
    #convert flattened vector (D,) to (1,D)
    query_vector = np.ascontiguousarray(
        image_embedding.detach().cpu().float().numpy()[np.newaxis,:], dtype="float32"
    )
    faiss.normalize_L2(query_vector)
    accuracies: dict[str, float] = {}
    for k in TOP_K:
        _, ids = faiss_index.search(query_vector, k)
        accuracies[f"top_{k}"] = float(
            any(records[i].step_number == image_record.step_number for i in ids[0] if i != -1)
        )
    return accuracies


def compute_accuracy_unit_scores_with_index(
    model_name: str,
    records: list[ImageRecord],
    instruction_indexes: dict[str, tuple[faiss.Index, list[ImageRecord]]], 
    checkpoint: str | None = None,
) -> dict[ImageRecord, dict[str, float]]:
    """
    Compute benchmark unit scores for given image records
    Return {query_record: {k: 1.0 if a top-k instruction has the same step else 0.0}}; storing is left to the caller.

    Query images without a stored embedding or whose build is not in the index are skipped
    """
    uncomputed_records = []
    embeddings = load_model_embeddings(model_name, checkpoint)
    scores: dict[ImageRecord, dict[str, float]] = {}
    for image_record in records:
        image_embedding = embeddings.get(image_record.key)
        if image_embedding is None:
            uncomputed_records.append(image_record)
            continue
        index = instruction_indexes.get(image_record.build_name)
        if index is None:
            uncomputed_records.append(image_record)
            continue
        accuracy_scores = _compute_top_k_accuracies(image_record,image_embedding, index)
        scores[image_record] = accuracy_scores
    if len(uncomputed_records) > 0:
        warnings.warn(f"""
        These records were uncomputed because no embedding 
        was found or index of corresponding build wasn't found
        {[record.image_name for record in uncomputed_records]}
        """)
    return scores
