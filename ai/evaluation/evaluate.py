"""
---------------------------------------------------------
VERINOTE

Module : Model Evaluation

Purpose:
Evaluates the trained MobileNetV2 model.

Author : Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.data.dataset_loader import get_datasets


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def main():

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = get_datasets()

    model = tf.keras.models.load_model(MODEL_PATH)

    predictions = model.predict(test_dataset)

    predicted_labels = (predictions > 0.5).astype(int).flatten()

    true_labels = np.concatenate([
        labels.numpy()
        for _, labels in test_dataset
    ]).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix\n")
    print(cm)

    print("\nClassification Report\n")
    print(report)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    with open(
        REPORT_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write("VERINOTE MODEL EVALUATION\n")
        file.write("=" * 60)
        file.write("\n\n")

        file.write(f"Accuracy  : {accuracy:.4f}\n")
        file.write(f"Precision : {precision:.4f}\n")
        file.write(f"Recall    : {recall:.4f}\n")
        file.write(f"F1 Score  : {f1:.4f}\n\n")

        file.write("Confusion Matrix\n")
        file.write(str(cm))
        file.write("\n\n")

        file.write("Classification Report\n\n")
        file.write(report)

    print("\nReports saved successfully.")

    print(f"\nConfusion Matrix : {REPORT_DIR / 'confusion_matrix.png'}")
    print(f"Classification Report : {REPORT_DIR / 'classification_report.txt'}")


if __name__ == "__main__":
    main()