"""Benchmark scores for the embeddings of a model.

The unit score of a query-image embedding is whether, among the instruction images ranked by
similarity (same model, same build), one of the top-k has the same step number as the query.
"""
from collections import defaultdict

from data.benchmarks_store import save_benchmarks
from data.embeddings_store import load_embeddings
from data.images_dto import ImageClass, ImageRecord
from data.query_processor import get_images_from_dataset
from data.similarities_store import load_similarities
import faiss
import numpy as np
import torch
import torch.nn.functional as F

TOP_K = [1, 3, 5]

def _compute_top_k_accuracies(
    image_record: ImageRecord,
    image_embedding: np.ndarray,
    index: tuple[faiss.Index, list[ImageRecord]],
) -> dict[str, float]:
    """Return {k: 1.0 if one of the k nearest instruction images has the query's step number else 0.0}."""
    faiss_index, records = index
    #faiss expects a (1, D) float32 batch; the index holds L2-normalized vectors (cosine)
    query = F.normalize(torch.from_numpy(image_embedding).float().reshape(1, -1), dim=-1)
    accuracies: dict[str, float] = {}
    for k in TOP_K:
        _, ids = faiss_index.search(query.numpy(), min(k, faiss_index.ntotal))
        accuracies[f"top_{k}"] = float(
            any(records[i].step_number == image_record.step_number for i in ids[0] if i != -1)
        )
    return accuracies


def compute_accuracy_unit_scores_with_index(
    model_name: str,
    records: list[ImageRecord],
    instruction_indexes: dict[str, tuple[faiss.Index, list[ImageRecord]]], 
    metric: str = "cosine",
) -> dict[ImageRecord, dict[str, float]]:
    """
    Compute benchmark unit scores for given image records
    Return {query_record: {k: 1.0 if a top-k instruction has the same step else 0.0}}.

    Query images without a stored embedding or without similarities are skipped.
    """
    #records whose embedding wasn't found on disk
    uncomputed_records = []


    embeddings = load_embeddings(model_name)
    scores: dict[ImageRecord, dict[str, float]] = {}
    for image_record in records:
        image_embedding = embeddings.get(image_record.key)
        if image_embedding is None:
            uncomputed_records.append(image_record)
            continue
        index = instruction_indexes.get(image_record.build_name)
        if index is None:
            raise ValueError(f"No index for {image_record.build_name} was found")
        accuracy_scores = _compute_top_k_accuracies(image_record,image_embedding, index)
        scores[image_record] = accuracy_scores
    save_benchmarks(model_name,metric,scores)
    return scores


def aggregate(scores: dict[ImageRecord, dict[int, float]]) -> dict[int, float]:
    """Mean top-k accuracy over all scored query images."""
    if not scores:
        return {}
    ks = next(iter(scores.values())).keys()
    return {k: sum(s[k] for s in scores.values()) / len(scores) for k in ks}
