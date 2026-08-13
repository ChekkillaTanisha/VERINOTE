"""
---------------------------------------------------------
VERINOTE

Module : Single Image Prediction

Purpose:
Predicts whether a currency note is Real or Fake
using the trained MobileNetV2 model.

Author : Tanisha Chekkilla
---------------------------------------------------------
"""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Fake",
    "Real"
]


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ---------------------------------------------------------
# PREDICT SINGLE IMAGE
# ---------------------------------------------------------

def predict(image_path):

    # Load image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )


    # Convert image to NumPy array
    image = tf.keras.utils.img_to_array(
        image
    )


    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    probabilities = model.predict(
        image,
        verbose=0
    )[0]


    # -----------------------------------------------------
    # SELECT CLASS USING HIGHEST PROBABILITY
    # -----------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = float(
        probabilities[predicted_index] * 100
    )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("VERINOTE PREDICTION")
    print("=" * 60)


    print(
        f"\nPrediction : {predicted_class}"
    )


    print(
        f"Confidence : {confidence:.2f}%"
    )


    print("\nClass Probabilities:")


    for index, class_name in enumerate(CLASS_NAMES):

        print(
            f"{class_name} : "
            f"{probabilities[index] * 100:.2f}%"
        )


    print("=" * 60)


    return predicted_class, confidence


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    image_path = input(
        "\nEnter image path : "
    ).strip()


    predict(
        image_path
    )