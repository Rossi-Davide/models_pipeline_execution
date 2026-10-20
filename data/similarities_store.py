"""Persistence of similarity scores between two embeddings of the same model and checkpoint.

Each "{model_name}-{checkpoint}" has its own file (just "{model_name}" when no checkpoint is given).
A score is identified by (metric, query ImageRecord.key, instruction ImageRecord.key),
so the same pair of images can hold different scores for different metrics.
"""
import csv
from pathlib import Path

from .images_store import ImageRecord

SIMILARITIES_DIR = Path(__file__).parent / "files"

ImageRecordKey = tuple[str, str, str | None]
SimilarityRecordKey = tuple[str, ImageRecordKey, ImageRecordKey]  # (metric, query key, instruction key)

_HEADER = ["metric", "query_folder", "query_image", "query_mod", "instr_folder", "instr_image", "instr_mod", "score"]


def _path(model_name: str, checkpoint: str | None) -> Path:
    #checkpoints can be paths like "openai/clip-vit-base-patch32"
    name = f"{model_name}-{checkpoint}".replace("/", "_") if checkpoint else model_name
    return SIMILARITIES_DIR / f"similarities_{name}.csv"


def _row(metric: str, q: ImageRecordKey, i: ImageRecordKey, score: float) -> list[str]:
    return [metric, q[0], q[1], q[2] or "", i[0], i[1], i[2] or "", repr(score)]

def load_similarities(model_name: str, checkpoint: str | None) -> dict[SimilarityRecordKey, float]:
    """Return {(metric, query_key, instruction_key): score} for a model and checkpoint."""
    path = _path(model_name, checkpoint)
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
    checkpoint: str | None = None,
) -> None:
    """Store {query_record: [(instruction_record, score)]}, merging with scores already saved.
        Each row is composed of (metric, query_image.key, other_image.key, similarity_score)
        The function assumes that the given similarity scores are calculated from the embeddings computed
        from the given model name and checkpoint
    """
    merged = load_similarities(model_name, checkpoint)
    for query, matches in similarities.items():
        for instruction, score in matches:
            merged[(metric, query.key, instruction.key)] = score

    SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(model_name, checkpoint), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for (m, q, i), score in merged.items():
            writer.writerow(_row(m, q, i, score))
