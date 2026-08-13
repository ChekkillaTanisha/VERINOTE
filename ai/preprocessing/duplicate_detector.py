"""
---------------------------------------------------------
VERINOTE

Module: Duplicate Detector

Purpose:
Detect duplicate images in the WORKING dataset
using MD5 hashing and generate a report.

Author: Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import hashlib

from ai.config.paths import WORKING_DATASET, REPORTS_DIR
from ai.config.constants import IMAGE_EXTENSIONS

def is_image_file(file_path: Path) -> bool:
    """
    Returns True if file is a supported image.
    """
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def calculate_md5(file_path: Path) -> str:
    """
    Calculates MD5 hash of an image.
    """

    md5 = hashlib.md5()

    with open(file_path, "rb") as image:

        while chunk := image.read(8192):
            md5.update(chunk)

    return md5.hexdigest()


def detect_duplicates():

    hashes = {}

    duplicates = []

    total_images = 0

    for image_path in WORKING_DATASET.rglob("*"):

        if not image_path.is_file():
            continue

        if not is_image_file(image_path):
            continue

        total_images += 1

        try:

            image_hash = calculate_md5(image_path)

            if image_hash in hashes:

                duplicates.append(
                    (
                        image_path,
                        hashes[image_hash]
                    )
                )

            else:

                hashes[image_hash] = image_path

        except Exception:

            pass

    return total_images, duplicates


def save_report(total_images, duplicates):

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_file = REPORTS_DIR / "duplicate_report.txt"

    with open(report_file, "w", encoding="utf-8") as report:

        report.write("=" * 60 + "\n")
        report.write("DUPLICATE IMAGE REPORT\n")
        report.write("=" * 60 + "\n\n")

        report.write(f"Total Images Checked : {total_images}\n")
        report.write(f"Duplicate Images Found : {len(duplicates)}\n\n")

        if duplicates:

            for duplicate, original in duplicates:

                report.write(f"Duplicate : {duplicate}\n")
                report.write(f"Original  : {original}\n")
                report.write("-" * 60 + "\n")

        else:

            report.write("No duplicate images found.\n")


def print_report(total_images, duplicates):

    print("=" * 60)
    print("DUPLICATE IMAGE REPORT")
    print("=" * 60)

    print(f"\nTotal Images Checked : {total_images}")
    print(f"Duplicate Images Found : {len(duplicates)}")

    print("\nReport Saved Successfully.")


def main():

    total_images, duplicates = detect_duplicates()

    save_report(total_images, duplicates)

    print_report(total_images, duplicates)


if __name__ == "__main__":
    main()
