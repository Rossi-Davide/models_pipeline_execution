import argparse
from collections import defaultdict

import faiss
import numpy as np
import torch

from data.images_dto import ImageClass, ImageRecord
from data.query_processor import get_images_for_build, get_images_from_dataset, store_embedding, store_similarities, list_builds, get_instructions_for_build, get_query_images_for_build
from pipeline_modelli import Model
from PIL import Image
from pathlib import Path
from typing import Iterable

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute image embeddings with a model.")
    parser.add_argument("--model", required=True, help=f"Model name, one of: {Model.available()}")
    parser.add_argument("--build", default=None, help="Build name; if omitted the entire dataset is used")
    return parser.parse_args()


def load_image_records(build: str | None) -> list[ImageRecord]:
    return get_images_for_build(build) if build else get_images_from_dataset()

def compute_embeddings_for_batch(model: Model, records: Iterable[ImageRecord]) -> list[tuple[ImageRecord, torch.Tensor]]:
    """Encode the images and return (record, embedding) pairs; storing is left to the caller."""
    results = []
    for record in records:
        path = Path(record.folder_path) / record.image_name
        with Image.open(path) as img:
            results.append((record, model.encode_image(img)))
    return results


def _to_matrix(embeddings: list[torch.Tensor]) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([e.detach().cpu().float().numpy().reshape(-1) for e in embeddings]), dtype="float32"
    )


def build_instruction_indexes(
    embedded: list[tuple[ImageRecord, torch.Tensor]],
) -> dict[str, tuple[faiss.Index, list[ImageRecord]]]:
    """
    Filters out embeddings not associated with building instruction images and 
    creates one cosine (normalized inner product) index per build.
    Each index contains the embeddings of the instruction pictures for a specific build

    Returns a dictionary {build_name: tuple}. Each tuple contains the index and the list of
    associated image records stored
    The records list is aligned with the index ids: id i -> records[i].
    """
    #groups records by build_name and filters out non instructions
    grouped: dict[str, list[tuple[ImageRecord, torch.Tensor]]] = defaultdict(list)
    for record, emb in embedded:
        if record.img_class == ImageClass.INSTRUCTION:
            grouped[record.build_name].append((record, emb))


    indexes = {}
    for build, items in grouped.items():
        #converts all embeddings of the manual in a matrix
        matrix = _to_matrix([e for _, e in items])
        #processes matrix and builds index from it
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        #stores records in the same order as the embeddings are stored in the index
        indexes[build] = (index, [r for r, _ in items])
    return indexes


def compute_similarities(
    instruction_indexes: dict[str, tuple[faiss.Index, list[ImageRecord]]],
    embedded: list[tuple[ImageRecord, torch.Tensor]],
) -> dict[ImageRecord, list[tuple[ImageRecord, float]]]:
    """For each query image, grab the 
    instruction_index of the corresponding build and measure 
    cosine similarity against all instruction images of its build
    Returns a dictionary {query_image: similarity values} containing a list
    of similarity values and the corresponding image record
    """
    results = {}
    for record, emb in embedded:
        #filter out non query images
        if record.img_class != ImageClass.QUERY_IMAGE or record.build_name not in instruction_indexes:
            continue
        #grab corresponding index
        index, instruction_records = instruction_indexes[record.build_name]
        q = _to_matrix([emb])
        faiss.normalize_L2(q)
        #compute scores
        scores, ids = index.search(q, index.ntotal)
        #build list of similarity values
        results[record] = [(instruction_records[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
    return results


def main() -> None:
    args = parse_args()
    model = Model(args.model)
    image_records = load_image_records(args.build)
    print(f"Loaded {model} and {len(image_records)} image records")

    embedded = compute_embeddings_for_batch(model, image_records)
    for record, embedding in embedded:
        #TODO: rework checkpoint paramenter
        store_embedding(record, model.name, embedding)

    indexes = build_instruction_indexes(embedded)
    similarities = compute_similarities(indexes, embedded)
    store_similarities(model.name, "cosine", similarities)
    print(f"Built {len(indexes)} index(es); computed similarities for {len(similarities)} query images")


if __name__ == "__main__":
    main()


