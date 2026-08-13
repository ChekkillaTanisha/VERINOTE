"""
---------------------------------------------------------
VERINOTE

Module : Single Image Prediction

Purpose:
Predicts whether a currency note is Real or Fake.

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


model = tf.keras.models.load_model(MODEL_PATH)


def predict(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image = tf.keras.utils.img_to_array(image)

    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    prediction = float(model.predict(image, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "Real"
        confidence = prediction * 100
    else:
        label = "Fake"
        confidence = (1 - prediction) * 100

    return {
        "prediction": label,
        "confidence": round(confidence, 2),
        "raw_prediction": round(prediction, 6)
    }


if __name__ == "__main__":

    image_path = input("\nEnter image path : ").strip()

    result = predict(image_path)

    print("=" * 60)
    print("VERINOTE PREDICTION")
    print("=" * 60)

    print(f"\nPrediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']}%")
    print(f"Raw Score  : {result['raw_prediction']}")