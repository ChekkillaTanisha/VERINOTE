"""
---------------------------------------------------------
VERINOTE
Module: Remove Corrupted Images

Purpose:
Checks every image in the working dataset and reports
corrupted images before preprocessing.

Author: Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
from PIL import Image

from config.paths import WORKING_DATASET
from config.constants import IMAGE_EXTENSIONS


def is_image_file(file_path: Path) -> bool:
    """
    Check whether the file is a supported image.
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def check_corrupted_images(dataset_path: Path) -> tuple[int, list[Path]]:
    """
    Checks all images inside the dataset.

    Args:
        dataset_path (Path): Dataset directory.

    Returns:
        tuple:
            Total images checked,
            List of corrupted image paths.
    """

    checked_images = 0
    corrupted_images = []

    for image_path in dataset_path.rglob("*"):

        if not image_path.is_file():
            continue

        if not is_image_file(image_path):
            continue

        checked_images += 1

        try:

            with Image.open(image_path) as image:
                image.verify()

        except Exception:

            corrupted_images.append(image_path)

    return checked_images, corrupted_images


def print_report(total_images: int, corrupted_images: list[Path]) -> None:
    """
    Prints dataset corruption report.
    """

    print("=" * 60)
    print("CORRUPTED IMAGE REPORT")
    print("=" * 60)

    print(f"\nImages Checked      : {total_images}")
    print(f"Corrupted Images   : {len(corrupted_images)}")

    if corrupted_images:

        print("\nCorrupted Files\n")

        for image in corrupted_images:
            print(image)

    else:

        print("\nNo corrupted images found.")

    print("\nDone.")


def main() -> None:
    """
    Entry point of the script.
    """

    total_images, corrupted_images = check_corrupted_images(
        WORKING_DATASET
    )

    print_report(total_images, corrupted_images)


if __name__ == "__main__":
    main()