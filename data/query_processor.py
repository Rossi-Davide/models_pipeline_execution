"""
Utility functions to call and handle write and read operations to disk files
"""
from pathlib import Path

from .images_store import ImageClass, ImageRecord, read_images, write_images

import numpy as np
import torch

from .benchmarks_store import save_benchmarks
from .embeddings_store import load_embeddings, save_embeddings_as_nparray
from .similarities_store import load_similarities, save_similarities

IMAGE_RECORDS_PATH = Path(__file__).parent / "files" / "image_records.csv"


def _load() -> list[ImageRecord]:
    if not IMAGE_RECORDS_PATH.exists() or IMAGE_RECORDS_PATH.stat().st_size == 0:
        return []
    return read_images(IMAGE_RECORDS_PATH)


def save_images(records: list[ImageRecord]) -> None:
    IMAGE_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_images(IMAGE_RECORDS_PATH, records)


def get_images_for_category(category: str) -> list[ImageRecord]:
    img_class = ImageClass(category)
    return [r for r in _load() if r.img_class == img_class]


def get_images_for_build(build_name: str) -> list[ImageRecord]:
    return [r for r in _load() if r.build_name == build_name]


def get_images_from_dataset() -> list[ImageRecord]:
    return _load()


def list_builds() -> list[str]:
    """Return the distinct build names in the dataset, in order of first appearance."""
    return list(dict.fromkeys(r.build_name for r in _load()))


def get_instructions_for_build(build_name: str) -> list[ImageRecord]:
    return [r for r in get_images_for_build(build_name) if r.img_class == ImageClass.INSTRUCTION]


def get_query_images_for_build(build_name: str) -> list[ImageRecord]:
    return [r for r in get_images_for_build(build_name) if r.img_class == ImageClass.QUERY_IMAGE]

def store_embeddings(
    embedded: list[tuple[ImageRecord, torch.Tensor]],
    model: str,
    checkpoint: str | None = None,
) -> None:
    """Persist the embeddings of many images in a single write, linked through image_record.key."""
    if not embedded:
        return
    records = [r for r, _ in embedded]
    # move tensor to cpu, convert to numpy array and flatten so that shape is (D,)
    # matrix has dimensions (N,D)
    matrix = np.stack([e.detach().cpu().float().numpy().flatten() for _, e in embedded])
    save_embeddings_as_nparray(model, checkpoint, records, matrix)


def store_similarities(
    model: str,
    metric: str,
    similarities: dict[ImageRecord, list[tuple[ImageRecord, float]]],
    checkpoint: str | None = None,
) -> None:
    """Persist query-instruction similarity scores computed from the given model's embeddings."""
    save_similarities(model, metric, similarities, checkpoint)

def load_model_embeddings(model: str, checkpoint: str | None = None) -> dict:
    """Load all stored embeddings of a model as torch tensors, keyed by image record key."""
    return {key: torch.from_numpy(vec).flatten() for key, vec in load_embeddings(model, checkpoint).items()}


def load_model_similarities(model: str, checkpoint: str | None = None):
    """Load all stored similarity scores of a model and checkpoint."""
    return load_similarities(model, checkpoint)


def store_benchmarks(
    model: str,
    metric: str,
    scores: dict[ImageRecord, dict[str, float]],
    checkpoint: str | None = None,
) -> None:
    """Persist benchmark scores computed for the given model and checkpoint."""
    save_benchmarks(model, metric, scores, checkpoint)


