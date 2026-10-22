"""Build image_records.csv by scanning <root>/<build>/<step>/<category>/<file>."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from data.images_store import ImageClass, ImageRecord, write_images

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INSTRUCTION_DIR = "instruction"
DEFAULT_OUTPUT = Path(__file__).parent / "data" / "files" / "image_records.csv"


def parse_tags(stem: str) -> list[str]:
    """Split the file stem on '-' and discard the trailing step number."""
    tokens = stem.split("-")
    # a leading '-' belongs to the first tag (e.g. "-90angle")
    if len(tokens) > 1 and tokens[0] == "":
        tokens = ["-" + tokens[1]] + tokens[2:]
    if tokens and tokens[-1].isdigit():
        tokens = tokens[:-1]
    elif tokens:
        tokens[-1] = re.sub(r"_\d+$", "", tokens[-1])
    return [t for t in tokens if t]


def scan(root: Path) -> list[ImageRecord]:
    records = []
    for path in sorted(root.glob("*/*/*/*")):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        build, step, category = path.relative_to(root).parts[:3]
        if not step.isdigit():
            print(f"warning: skipping {path} (non-numeric step folder)", file=sys.stderr)
            continue
        records.append(
            ImageRecord(
                folder_path=path.parent.as_posix(),
                build_name=build,
                image_name=path.name,
                step_number=int(step),
                img_class=ImageClass.INSTRUCTION if category == INSTRUCTION_DIR else ImageClass.QUERY_IMAGE,
                tags=parse_tags(path.stem),
            )
        )
    return sorted(records, key=lambda r: (r.build_name, r.step_number, r.folder_path, r.image_name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dataset root folder")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"{args.root} is not a directory")

    records = scan(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_images(args.output, records)
    print(f"Wrote {len(records)} records to {args.output}")
    for label, counter in (
        ("class", Counter(r.img_class.value for r in records)),
        ("build", Counter(r.build_name for r in records)),
    ):
        print(f"  by {label}: {dict(counter)}")


if __name__ == "__main__":
    main()
