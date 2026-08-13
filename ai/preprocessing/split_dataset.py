"""
---------------------------------------------------------
VERINOTE

Module: Dataset Splitter

Purpose:
Splits the cleaned working dataset into
Train, Validation and Test datasets.

Author: Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import random
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.config.paths import WORKING_DATASET, PROCESSED_DATASET
from ai.config.settings import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)
from ai.config.constants import IMAGE_EXTENSIONS


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def copy_images(images: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(image, destination / image.name)


def split_folder(folder: Path) -> tuple[list[Path], list[Path], list[Path]]:
    images = [img for img in folder.iterdir() if img.is_file() and is_image_file(img)]

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_SPLIT)
    validation_end = train_end + int(total * VALIDATION_SPLIT)

    train = images[:train_end]
    validation = images[train_end:validation_end]
    test = images[validation_end:]

    return train, validation, test


def main():

    random.seed(RANDOM_SEED)

    if PROCESSED_DATASET.exists():
        shutil.rmtree(PROCESSED_DATASET)

    classes = ["Real", "Fake"]

    train_count = 0
    validation_count = 0
    test_count = 0

    for class_name in classes:

        class_path = WORKING_DATASET / class_name

        if not class_path.exists():
            continue

        for denomination in sorted(class_path.iterdir()):

            if not denomination.is_dir():
                continue

            train, validation, test = split_folder(denomination)

            copy_images(
                train,
                PROCESSED_DATASET /
                "train" /
                class_name /
                denomination.name,
            )

            copy_images(
                validation,
                PROCESSED_DATASET /
                "validation" /
                class_name /
                denomination.name,
            )

            copy_images(
                test,
                PROCESSED_DATASET /
                "test" /
                class_name /
                denomination.name,
            )

            train_count += len(train)
            validation_count += len(validation)
            test_count += len(test)

    print("=" * 60)
    print("DATASET SPLIT REPORT")
    print("=" * 60)

    print(f"\nTraining Images   : {train_count}")
    print(f"Validation Images : {validation_count}")
    print(f"Testing Images    : {test_count}")

    print(
        f"\nTotal Images      : "
        f"{train_count + validation_count + test_count}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()