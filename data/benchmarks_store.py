"""Persistence of per-embedding benchmark (top-k) scores, keyed by (model, metric, query ImageRecord.key)."""
import csv
from pathlib import Path

from .images_store import ImageRecord

BENCHMARKS_DIR = Path(__file__).parent / "files"

ImageRecordKey = tuple[str, str, str | None]

_HEADER = ["metric", "folder", "image", "mod", "benchmark_metric", "unit_score"]


def _path(model_name: str) -> Path:
    return BENCHMARKS_DIR / f"benchmarks_{model_name}.csv"


def load_benchmarks(model_name: str) -> dict[tuple[str, ImageRecordKey], dict[str, float]]:
    """Return {(metric, query_key): {benchmark_metric: unit_score}} for a model."""
    path = _path(model_name)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[tuple[str, ImageRecordKey], dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["metric"], (r["folder"], r["image"], r["mod"] or None))
            out.setdefault(key, {})[r["benchmark_metric"]] = float(r["unit_score"])
    return out


def save_benchmarks(model_name: str, metric: str, scores: dict[ImageRecord, dict[str, float]]) -> None:
    """Store {query_record: {benchmark_metric: unit_score}}, merging with scores already saved."""
    merged = load_benchmarks(model_name)
    for record, by_k in scores.items():
        #set new scores dictionary for given entry, removes all previous scores for {metric,image_record}!!
        merged.setdefault((metric, record.key), {}).update(by_k)

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(model_name), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for (m, k), by_k in merged.items():
            for top_k, score in sorted(by_k.items()):
                writer.writerow([m, k[0], k[1], k[2] or "", top_k, repr(score)])
