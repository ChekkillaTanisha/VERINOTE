"""
---------------------------------------------------------
VERINOTE

Module : Misclassified Image Analysis

Purpose:
Identifies incorrectly classified test images so that
model and dataset errors can be analyzed.

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


from ai.data.dataset_loader import get_datasets


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.keras"

CLASS_NAMES = [
    "Fake",
    "Real"
]


def main():

    print("=" * 60)
    print("VERINOTE MISCLASSIFIED IMAGE ANALYSIS")
    print("=" * 60)

    (
        _,
        _,
        test_dataset,
        class_names
    ) = get_datasets()


    print("\nClass Names:")
    print(class_names)


    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    # -----------------------------------------------------
    # GET TEST IMAGES, LABELS AND FILE PATHS
    # -----------------------------------------------------

    test_directory = (
        PROJECT_ROOT
        / "ai"
        / "datasets"
        / "processed"
        / "test"
    )


    image_paths = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_directory = (
            test_directory
            / class_name
        )

        for image_path in sorted(
            class_directory.rglob("*")
        ):

            if image_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:

                image_paths.append(
                    (
                        str(image_path),
                        class_index
                    )
                )


    print(
        f"\nTest images found: {len(image_paths)}"
    )


    # -----------------------------------------------------
    # PREDICT EVERY TEST IMAGE
    # -----------------------------------------------------

    wrong_predictions = []


    for image_path, true_index in image_paths:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=(224, 224)
        )


        image = tf.keras.utils.img_to_array(
            image
        )


        image = np.expand_dims(
            image,
            axis=0
        )


        prediction = model.predict(
            image,
            verbose=0
        )[0]


        predicted_index = int(
            np.argmax(prediction)
        )


        confidence = float(
            prediction[predicted_index]
        )


        # -------------------------------------------------
        # ONLY STORE WRONG PREDICTIONS
        # -------------------------------------------------

        if predicted_index != true_index:

            wrong_predictions.append(
                {
                    "path": image_path,
                    "actual": CLASS_NAMES[true_index],
                    "predicted": CLASS_NAMES[predicted_index],
                    "confidence": confidence * 100,
                    "fake_probability": prediction[0] * 100,
                    "real_probability": prediction[1] * 100
                }
            )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("MISCLASSIFIED IMAGES")
    print("=" * 60)


    print(
        f"\nTotal Misclassified: "
        f"{len(wrong_predictions)}"
    )


    fake_to_real = 0
    real_to_fake = 0


    for item in wrong_predictions:

        if (
            item["actual"] == "Fake"
            and
            item["predicted"] == "Real"
        ):

            fake_to_real += 1


        elif (
            item["actual"] == "Real"
            and
            item["predicted"] == "Fake"
        ):

            real_to_fake += 1


    print(
        f"\nFake -> Real : {fake_to_real}"
    )

    print(
        f"Real -> Fake : {real_to_fake}"
    )


    print("\n" + "-" * 60)


    for number, item in enumerate(
        wrong_predictions,
        start=1
    ):

        print(
            f"\n[{number}]"
        )

        print(
            f"Actual      : {item['actual']}"
        )

        print(
            f"Predicted   : {item['predicted']}"
        )

        print(
            f"Confidence  : "
            f"{item['confidence']:.2f}%"
        )

        print(
            f"Fake        : "
            f"{item['fake_probability']:.2f}%"
        )

        print(
            f"Real        : "
            f"{item['real_probability']:.2f}%"
        )

        print(
            f"Image Path  : {item['path']}"
        )


    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()