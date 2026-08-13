import os
from collections import defaultdict

# ==============================
# Dataset Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATASET = os.path.join(BASE_DIR, "datasets", "raw")

REAL_FAKE_DATASET = os.path.join(
    RAW_DATASET,
    "Indian_Currency_Real_vs_Fake"
)

GENUINE_DATASET = os.path.join(
    RAW_DATASET,
    "Indian_Currency_Dataset"
)

# Supported Image Extensions
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ==============================
# Function
# ==============================

def count_images(folder):

    total = 0
    denomination_count = defaultdict(int)

    for denomination in sorted(os.listdir(folder)):

        denomination_path = os.path.join(folder, denomination)

        if not os.path.isdir(denomination_path):
            continue

        count = 0

        for file in os.listdir(denomination_path):

            if file.lower().endswith(IMAGE_EXTENSIONS):
                count += 1

        denomination_count[denomination] = count
        total += count

    return total, denomination_count

# ==============================
# REAL
# ==============================

real_path = os.path.join(REAL_FAKE_DATASET, "Real")

real_total, real_data = count_images(real_path)

# ==============================
# FAKE
# ==============================

fake_path = os.path.join(REAL_FAKE_DATASET, "Fake")

fake_total, fake_data = count_images(fake_path)

# ==============================
# GENUINE DATASET
# ==============================

genuine_total, genuine_data = count_images(GENUINE_DATASET)

# ==============================
# PRINT REPORT
# ==============================

print("=" * 50)
print("VERINOTE DATASET REPORT")
print("=" * 50)

print("\nREAL NOTES")

for d, c in real_data.items():
    print(f"₹{d} : {c}")

print(f"\nTotal Real Images : {real_total}")

print("\n" + "-" * 50)

print("\nFAKE NOTES")

for d, c in fake_data.items():
    print(f"₹{d} : {c}")

print(f"\nTotal Fake Images : {fake_total}")

print("\n" + "-" * 50)

print("\nGENUINE DATASET")

for d, c in genuine_data.items():
    print(f"₹{d} : {c}")

print(f"\nTotal Genuine Images : {genuine_total}")

print("\n" + "=" * 50)

print("OVERALL TOTAL")

print(real_total + fake_total + genuine_total)

print("=" * 50)