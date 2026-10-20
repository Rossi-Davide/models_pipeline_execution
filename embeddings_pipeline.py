import argparse

import torch

from data.images_store import ImageRecord, ImageClass
from data.query_processor import get_images_for_build, get_images_from_dataset, store_embeddings, store_similarities
from pipeline_modelli import Model
from similarity_index import build_flat_indexes_per_build, compute_cosine_similarities_from_embeddings
from benchmark import compute_accuracy_unit_scores_with_index
from PIL import Image
from pathlib import Path
from typing import Iterable

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute image embeddings with a model.")
    parser.add_argument("--model", required=True, help=f"Model name, one of: {Model.available()}")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint; defaults to the model's default checkpoint")
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
    #load model
    model = Model(args.model, checkpoint=args.checkpoint)
    #load image records
    query_records = load_image_records(args.build)
    print(f"Loaded {model} and {len(query_records)} image records")
    #compute and store embeddings
    embedded = compute_embeddings_for_batch(model, query_records)
    print(f"Embeddings computed for {len(embedded)}")
    store_embeddings(embedded, model.name, model.checkpoint)
    print("Embeddings stored successfully on file")
    #create FAISS indexes from embeddings for faster lookup
    instruction_embeds = [(record, embedding)  for (record, embedding) in embedded if record.img_class == ImageClass.INSTRUCTION]
    query_embeds = [(record, embedding)  for (record, embedding) in embedded if record.img_class == ImageClass.QUERY_IMAGE]
    instruction_indexes = build_flat_indexes_per_build(instruction_embeds)
    print(f"Instruction indexes built for images")
    for build in instruction_indexes.keys():
        print(f"Build:{build} with {len(instruction_indexes[build][1])} images")
    #compute cosine similarities of query images against instruction images of the same build
    print("Computing cosine similarities")
    similarities = compute_cosine_similarities_from_embeddings(instruction_indexes, query_embeds)
    print("Storing cosine similarities")
    store_similarities(model.name, "cosine", similarities, model.checkpoint)
    print(f"Computed similarities for {len(similarities)} query images")
    # compute accuracy unit scores
    query_records = [record for record, _ in query_embeds]
    #unit scores might vary based on benchmark metric, embeddings and similarity metric
    compute_accuracy_unit_scores_with_index(model.name, query_records, instruction_indexes, checkpoint=model.checkpoint)

if __name__ == "__main__":
    main()


