import argparse

from data.images_dto import ImageRecord
from data.query_processor import get_images_for_build, get_images_from_dataset, store_embedding
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

def compute_embeddings_for_batch(model: Model, records: Iterable[ImageRecord]):
    #feed images to the model
    for record  in records:
        path = Path(record.folder_path) / record.image_name
        img = Image.open(path)
        img_embedding = model.encode_image(img)
        #TODO: rework checkpoint paramenter
        store_embedding(record, model.name, img_embedding)

    

def main() -> None:
    #Loading model and image records
    args = parse_args()
    model_name = args.model
    model = Model(model_name)
    image_records = load_image_records(args.build)
    print(f"Loaded {model} and {len(image_records)} image records")
    compute_embeddings_for_batch(model, image_records)
    compute_similarities_for_build()



if __name__ == "__main__":
    main()


