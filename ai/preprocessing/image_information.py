import os
from collections import Counter
from PIL import Image

# ======================================================
# PATH
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "Indian_Currency_Real_vs_Fake"
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ======================================================

formats = Counter()
sizes = Counter()
modes = Counter()

total_images = 0

# ======================================================

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if not file.lower().endswith(IMAGE_EXTENSIONS):
            continue

        image_path = os.path.join(root, file)

        try:

            with Image.open(image_path) as img:

                total_images += 1

                formats[img.format] += 1
                sizes[img.size] += 1
                modes[img.mode] += 1

        except Exception:
            pass

# ======================================================

print("="*60)
print("IMAGE INFORMATION REPORT")
print("="*60)

print(f"\nTotal Images : {total_images}")

print("\nIMAGE FORMATS")

for k,v in formats.items():
    print(f"{k} : {v}")

print("\nIMAGE MODES")

for k,v in modes.items():
    print(f"{k} : {v}")

print("\nTOP 10 IMAGE SIZES")

for size,count in sizes.most_common(10):
    print(f"{size} : {count}")

print("\nDone.")