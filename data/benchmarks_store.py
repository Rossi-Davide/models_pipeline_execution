"""Persistence of per-embedding benchmark (top-k) scores, keyed by (similarity_metric, query ImageRecord.key, benchmark_metric).
    Each "{model_name}-{checkpoint}" has its own file (just "{model_name}" when no checkpoint is given)
"""
import csv
import pandas as pd
from pathlib import Path

from .images_store import ImageRecord

BENCHMARKS_DIR = Path(__file__).parent / "files"

ImageRecordKey = tuple[str, str, str | None]

_HEADER = ["metric", "folder", "image", "mod", "benchmark_metric", "unit_score"]


def _path(model_name: str, checkpoint: str | None) -> Path:
    #checkpoints can be paths like "openai/clip-vit-base-patch32"
    name = f"{model_name}-{checkpoint}".replace("/", "_") if checkpoint else model_name
    return BENCHMARKS_DIR / f"benchmarks_{name}.csv"

def load_benchmarks(model_name: str, checkpoint: str | None) -> dict[tuple[str, ImageRecordKey], dict[str, float]]:
    """Return {(metric, query_key): {benchmark_metric: unit_score}} for a model and checkpoint."""
    path = _path(model_name, checkpoint)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[tuple[str, ImageRecordKey], dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["metric"], (r["folder"], r["image"], r["mod"] or None))
            out.setdefault(key, {})[r["benchmark_metric"]] = float(r["unit_score"])
    return out


def save_benchmarks(
    model_name: str,
    metric: str,
    scores: dict[ImageRecord, dict[str, float]],
    checkpoint: str | None = None,
) -> None:
    """Store {query_record: {benchmark_metric: unit_score}}, merging with scores already saved."""
    merged = load_benchmarks(model_name, checkpoint)
    for record, benchmark_scores in scores.items():
        merged.setdefault((metric, record.key), {}).update(benchmark_scores)

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(model_name, checkpoint), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HEADER)
        for (m, k), by_k in merged.items():
            for top_k, score in sorted(by_k.items()):
                writer.writerow([m, k[0], k[1], k[2] or "", top_k, repr(score)])

def load_benchmarks_as_df(model_name: str, checkpoint: str | None) -> pd.DataFrame:
    path = _path(model_name, checkpoint)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, sep=',', header=0)
    return df
