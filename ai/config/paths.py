"""
-----------------------------------------
VERINOTE
Project Paths
-----------------------------------------
"""

from pathlib import Path

# Root folder (VERINOTE)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# AI folder
AI_ROOT = PROJECT_ROOT / "ai"

# Dataset folders
RAW_DATASET = AI_ROOT / "datasets" / "raw" / "Indian_Currency_Real_vs_Fake"

WORKING_DATASET = AI_ROOT / "datasets" / "working" / "Indian_Currency_Real_vs_Fake"

PROCESSED_DATASET = AI_ROOT / "datasets" / "processed"

DUPLICATES_REVIEW = AI_ROOT / "datasets" / "duplicates_review"

# Models
MODELS_DIR = AI_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "docs" / "reports"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"