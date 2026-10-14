"""Similarity metrics between embeddings."""
from __future__ import annotations

from typing import Callable, Dict, List

import torch
import torch.nn.functional as F


#TODO: figure out output dimension and shape
# Each metric takes two tensors of shape (..., D) and returns similarity over the last dim.
_METRICS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cosine": lambda a, b: F.cosine_similarity(a, b, dim=-1),
}


def available_metrics() -> List[str]:
    return sorted(_METRICS)


def similarity(metric: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the similarity of two embeddings using the named metric."""
    try:
        fn = _METRICS[metric]
    except KeyError:
        raise ValueError(f"Unknown metric '{metric}'. Available: {available_metrics()}") from None
    return fn(a, b)
