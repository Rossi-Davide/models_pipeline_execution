"""Persistence of similarity scores between two embeddings of the same model.

A score is identified by (model, metric, query ImageRecord.key, instruction ImageRecord.key),
so the same pair of images can hold different scores for different models or metrics.
"""
import csv
from pathlib import Path

from .images_dto import ImageRecord

SIMILARITIES_DIR = Path(__file__).parent / "files"

Key = tuple[str, str, str | None]
Pair = tuple[str, Key, Key]  # (metric, query key, instruction key)

_HEADER = ["metric", "query_folder", "query_image", "query_mod", "instr_folder", "instr_image", "instr_mod", "score"]


def _path(model_name: str) -> Path:
    return SIMILARITIES_DIR / f"similarities_{model_name}.csv"


def _row(metric: str, q: Key, i: Key, score: float) -> list[str]:
    # None modification is stored as an empty string
    return [metric, q[0], q[1], q[2] or "", i[0], i[1], i[2] or "", repr(score)]


def load_similarities(model_name: str) -> dict[Pair, float]:
    """Return {(metric, query_key, instruction_key): score} for a model."""
    path = _path(model_name)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {
            (r["metric"], (r["query_folder"], r["query_image"], r["query_mod"] or None),
             (r["instr_folder"], r["instr_image"], r["instr_mod"] or None)): float(r["score"])
            for r in csv.DictReader(f)
        }


def save_similarities(
    model_name: str,
    metric: str,
    similarities: dict[ImageRecord, list[tuple[ImageRecord, float]]],
) -> None:
    """Store {query_record: [(instruction_record, score), ...]}, merging with scores already saved."""
    merged = load_similarities(model_name)
    for query, matches in similarities.items():
        for instruction, score in matches:
            merged[(metric, query.key, instruction.key)] = score

    SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(model_name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for (m, q, i), score in merged.items():
            writer.writerow(_row(m, q, i, score))
