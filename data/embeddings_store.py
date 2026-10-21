"""Persistence of embeddings, linked to images through ImageRecord.key (not row order).
    Module stores a .npz file per "{model_name}-{checkpoint}" (just "{model_name}" when no checkpoint
    is given) with {ImageRecord.key: embedding}
"""
from pathlib import Path

import numpy as np

from .images_store import ImageRecord

EMBEDDINGS_DIR = Path(__file__).parent / "files"

ImageRecordKey = tuple[str, str, str | None]


def _path(model_name: str, checkpoint: str | None) -> Path:
    #checkpoints can be paths like "openai/clip-vit-base-patch32"
    name = f"{model_name}-{checkpoint}".replace("/", "_") if checkpoint else model_name
    return EMBEDDINGS_DIR / f"embeddings_{name}.npz"


def save_embeddings_as_nparray(
    model_name: str,
    checkpoint: str | None,
    records: list[ImageRecord],
    embeddings: np.ndarray,
) -> None:
    """Store embeddings (one row per record) for a model and checkpoint, merging with any already saved."""
    if len(records) != len(embeddings):
        raise ValueError(f"{len(records)} records but {len(embeddings)} embeddings")
    merged = load_embeddings(model_name, checkpoint)
    merged.update({r.key: np.asarray(e) for r, e in zip(records, embeddings)})

    keys = list(merged)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _path(model_name, checkpoint),
        # None modification is stored as an empty string
        keys=np.array([[k[0], k[1], k[2] or ""] for k in keys], dtype=str).reshape(-1, 3),
        embeddings=np.stack([merged[k] for k in keys]) if keys else np.empty((0, 0)),
    )

def load_embeddings(model_name: str, checkpoint: str | None = None) -> dict[ImageRecordKey, np.ndarray]:
    """Return {ImageRecord.key: embedding} for a model and checkpoint."""
    path = _path(model_name, checkpoint)
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {
            (k[0], k[1], k[2] or None): e
            for k, e in zip(data["keys"].tolist(), data["embeddings"])
        }