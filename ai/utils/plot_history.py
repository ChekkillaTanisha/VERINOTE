"""
---------------------------------------------------------
VERINOTE

Module : Training History Plot

Purpose:
Plots training and validation graphs.

Author : Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import matplotlib.pyplot as plt


def plot_history(history):

    reports_dir = Path("reports/graphs")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(reports_dir / "accuracy.png")
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(reports_dir / "loss.png")
    plt.close()

    print("\nTraining graphs saved successfully.")