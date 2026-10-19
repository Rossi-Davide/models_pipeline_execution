"""Persistence of embeddings, linked to images through ImageRecord.key (not row order).
    Module stores a .npz file per model with {ImageRecord.key: embedding}
"""
from pathlib import Path

import numpy as np

from .images_store import ImageRecord

EMBEDDINGS_DIR = Path(__file__).parent / "files"

ImageRecordKey = tuple[str, str, str | None]


def _path(model_name: str) -> Path:
    return EMBEDDINGS_DIR / f"embeddings_{model_name}.npz"


def save_embeddings(
    model_name: str,
    checkpoint: str | None,
    records: list[ImageRecord],
    embeddings: np.ndarray,
) -> None:
    """Store embeddings (one row per record) for a model, merging with any already saved."""
    if len(records) != len(embeddings):
        raise ValueError(f"{len(records)} records but {len(embeddings)} embeddings")
    stored_checkpoint, merged = _read(model_name, checkpoint)
    merged.update({r.key: np.asarray(e) for r, e in zip(records, embeddings)})

    keys = list(merged)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _path(model_name),
        checkpoint=np.array(checkpoint or stored_checkpoint),
        # None modification is stored as an empty string
        keys=np.array([[k[0], k[1], k[2] or ""] for k in keys], dtype=str).reshape(-1, 3),
        embeddings=np.stack([merged[k] for k in keys]) if keys else np.empty((0, 0)),
    )


def _read(model_name: str, checkpoint: str | None) -> tuple[str, dict[ImageRecordKey, np.ndarray]]:
    path = _path(model_name)
    if not path.exists():
        return "", {}
    data = np.load(path)
    stored = str(data["checkpoint"])
    if checkpoint and stored and checkpoint != stored:
        raise ValueError(f"Stored embeddings use checkpoint '{stored}', not '{checkpoint}'")
    return stored, {
        (k[0], k[1], k[2] or None): e
        for k, e in zip(data["keys"].tolist(), data["embeddings"])
    }


def load_embeddings(model_name: str, checkpoint: str | None = None) -> dict[ImageRecordKey, np.ndarray]:
    """Return {ImageRecord.key: embedding}. If checkpoint is given it must match the stored one."""
    return _read(model_name, checkpoint)[1]


def get_embedding(model_name: str, record: ImageRecord) -> np.ndarray | None:
    return load_embeddings(model_name).get(record.key)
