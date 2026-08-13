"""
---------------------------------------------------------
VERINOTE

Module: Clean Duplicate Images

Purpose:
Moves duplicate images from the working dataset
to the duplicates_review folder without deleting them.

Author: Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.config.paths import DUPLICATES_REVIEW, REPORTS_DIR


REPORT_FILE = REPORTS_DIR / "duplicate_report.txt"


def move_duplicates() -> int:
    """
    Move duplicate images to duplicates_review.

    Returns:
        int: Number of moved images.
    """

    if not REPORT_FILE.exists():
        print("Duplicate report not found.")
        return 0

    DUPLICATES_REVIEW.mkdir(parents=True, exist_ok=True)

    moved = 0

    with open(REPORT_FILE, "r", encoding="utf-8") as report:

        for line in report:

            if line.startswith("Duplicate :"):

                duplicate_path = Path(
                    line.replace("Duplicate :", "").strip()
                )

                if duplicate_path.exists():

                    destination = DUPLICATES_REVIEW / duplicate_path.name

                    shutil.move(
                        duplicate_path,
                        destination
                    )

                    moved += 1

    return moved


def main():

    moved = move_duplicates()

    print("=" * 60)
    print("DUPLICATE CLEANING REPORT")
    print("=" * 60)

    print(f"\nDuplicate Images Moved : {moved}")

    print("\nDone.")


if __name__ == "__main__":
    main()