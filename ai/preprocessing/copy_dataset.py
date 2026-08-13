import os
import shutil

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "raw",
    "Indian_Currency_Real_vs_Fake"
)

WORKING_DATASET = os.path.join(
    BASE_DIR,
    "datasets",
    "working",
    "Indian_Currency_Real_vs_Fake"
)

# ======================================================
# COPY DATASET
# ======================================================

if os.path.exists(WORKING_DATASET):
    print("Working dataset already exists.")
    print("Delete it manually if you want to create a fresh copy.")

else:
    print("Copying dataset...")

    shutil.copytree(RAW_DATASET, WORKING_DATASET)

    print("Dataset copied successfully!")

print("\nDone.")