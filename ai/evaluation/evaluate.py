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

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ai.data.dataset_loader import get_datasets


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"


def main():

    # ---------------------------------------------------------
    # LOAD DATASETS
    # ---------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        class_names
    ) = get_datasets()


    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # ---------------------------------------------------------
    # MODEL PREDICTIONS
    # ---------------------------------------------------------

    predictions = model.predict(
        test_dataset,
        verbose=1
    )


    # ---------------------------------------------------------
    # CONVERT SOFTMAX OUTPUT TO CLASS INDEX
    # ---------------------------------------------------------

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )


    # ---------------------------------------------------------
    # GET TRUE LABELS
    # ---------------------------------------------------------

    true_labels = np.concatenate([

        np.argmax(labels.numpy(), axis=1)

        for _, labels in test_dataset

    ])


    # ---------------------------------------------------------
    # DISPLAY EVALUATION
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VERINOTE MODEL EVALUATION")
    print("=" * 60)


    print("\nClass Names:")
    print(class_names)


    print("\nClass Mapping:")

    for index, class_name in enumerate(class_names):

        print(
            f"{index} -> {class_name}"
        )


    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )


    print("\nModel Performance:")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )


    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    print("\nConfusion Matrix\n")

    matrix = confusion_matrix(
        true_labels,
        predicted_labels
    )

    print(matrix)


    # ---------------------------------------------------------
    # CLASSIFICATION REPORT
    # ---------------------------------------------------------

    print("\nClassification Report\n")

    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=class_names,
            zero_division=0
        )
    )


    # ---------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # ---------------------------------------------------------

    print("\nPrediction Distribution:")

    for index, class_name in enumerate(class_names):

        count = np.sum(
            predicted_labels == index
        )

        print(
            f"{class_name}: {count}"
        )


    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()