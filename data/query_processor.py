"""
Utility functions to call and handle write and read operations to disk files
"""
from pathlib import Path

from .images_dto import ImageClass, ImageRecord, read_images, write_images

IMAGE_RECORDS_PATH = Path(__file__).parent / "files" / "image_records.csv"


def _load() -> list[ImageRecord]:
    if not IMAGE_RECORDS_PATH.exists() or IMAGE_RECORDS_PATH.stat().st_size == 0:
        return []
    return read_images(IMAGE_RECORDS_PATH)


def save_images(records: list[ImageRecord]) -> None:
    IMAGE_RECORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_images(IMAGE_RECORDS_PATH, records)


def get_images_for_category(category: str) -> list[ImageRecord]:
    img_class = ImageClass(category)
    return [r for r in _load() if r.img_class == img_class]


def get_images_for_build(build_name: str) -> list[ImageRecord]:
    return [r for r in _load() if r.build_name == build_name]


def get_images_from_dataset() -> list[ImageRecord]:
    return _load()


def get_instructions_for_build(build_name: str) -> list[ImageRecord]:
    return [r for r in get_images_for_build(build_name) if r.img_class == ImageClass.INSTRUCTION]


def get_query_images_for_build(build_name: str) -> list[ImageRecord]:
    return [r for r in get_images_for_build(build_name) if r.img_class == ImageClass.QUERY_IMAGE]

