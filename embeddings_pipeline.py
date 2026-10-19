import argparse

import torch

from data.images_dto import ImageRecord
from data.query_processor import get_images_for_build, get_images_from_dataset, store_embedding, store_similarities, list_builds, get_instructions_for_build, get_query_images_for_build
from pipeline_modelli import Model
from similarity_index import build_instruction_indexes, compute_similarities
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


