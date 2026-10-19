"""
Utility functions to call and handle write and read operations to disk files
"""
from pathlib import Path

from .images_store import ImageClass, ImageRecord, read_images, write_images

import torch

from .benchmarks_store import save_benchmarks
from .embeddings_store import load_embeddings, save_embeddings
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

def store_embedding(
    image_record: ImageRecord,
    model: str,
    embedding: torch.Tensor,
    checkpoint: str | None = None,
) -> None:
    """Persist the embedding of one image for the given model, linked through image_record.key."""
    vector = embedding.detach().cpu().float().numpy().reshape(1, -1)
    save_embeddings(model, checkpoint, [image_record], vector)

def store_similarities(
    model: str,
    metric: str,
    similarities: dict[ImageRecord, list[tuple[ImageRecord, float]]],
) -> None:
    """Persist query-instruction similarity scores computed from the given model's embeddings."""
    save_similarities(model, metric, similarities)

def load_model_embeddings(model: str, checkpoint: str | None = None) -> dict:
    """Load all stored embeddings of a model as torch tensors, keyed by image record key."""
    return {key: torch.from_numpy(vec) for key, vec in load_embeddings(model, checkpoint).items()}


def load_model_similarities(model: str):
    """Load all stored similarity scores of a model."""
    return load_similarities(model)


def store_benchmarks(model: str, metric: str, scores: dict[ImageRecord, dict[str, float]]) -> None:
    """Persist benchmark scores computed for the given model."""
    save_benchmarks(model, metric, scores)


