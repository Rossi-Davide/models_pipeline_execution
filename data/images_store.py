"""Data transfer objects for the images file."""
from __future__ import annotations

import csv
from enum import Enum
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union


class ImageClass(str, Enum):
    QUERY_IMAGE = "query_image"
    INSTRUCTION = "instruction"


@dataclass(frozen=True)
class ImageRecord:
    folder_path: str
    build_name: str
    image_name: str
    step_number: int
    img_class: ImageClass
    tags: list[str] = field(hash=False, compare=False)
    modification: Optional[str] = None


    @property
    def key(self) -> Tuple[str, str, Optional[str]]:
        """Unique identifier of an image (replaces the DB id)."""
        return (self.folder_path, self.image_name, self.modification)


_COLUMNS = [f.name for f in fields(ImageRecord)]


def write_images(path: Union[str, Path], records: Iterable[ImageRecord]) -> None:
    """Write records to a tab-separated txt file (None modification -> empty field)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_COLUMNS)
        for r in records:
            tags_string = "|".join(r.tags)
            writer.writerow([r.folder_path, r.build_name, r.image_name, r.step_number, r.img_class.value,tags_string,r.modification or ""])


def read_images(path: Union[str, Path]) -> List[ImageRecord]:
    """Read records written by :func:`write_images`."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            ImageRecord(
                folder_path=row["folder_path"],
                build_name=row["build_name"],
                image_name=row["image_name"],
                step_number=int(row["step_number"]),
                img_class=ImageClass(row["img_class"]),
                tags=[tag for tag in row.get("tags","").split('|') if tag != ""],
                modification=row["modification"] or None,
            )
            for row in csv.DictReader(f)
        ]
