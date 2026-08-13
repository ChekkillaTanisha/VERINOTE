"""
---------------------------------------------------------
VERINOTE

Module : Model Training

Purpose:
Compiles and trains the MobileNetV2 model.

Author : Tanisha Chekkilla
---------------------------------------------------------
"""

from ai.utils.plot_history import plot_history
from pathlib import Path
import sys
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.models.mobilenet import build_model
from ai.data.dataset_loader import get_datasets
from ai.config.training import LEARNING_RATE, EPOCHS, PATIENCE


MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


(
    train_dataset,
    validation_dataset,
    test_dataset,
    class_names
) = get_datasets()


model = build_model()


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=["accuracy"]

)


checkpoint = tf.keras.callbacks.ModelCheckpoint(

    filepath=MODEL_DIR / "best_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)


early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=PATIENCE,

    restore_best_weights=True,

    verbose=1

)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping
    ]

)
plot_history(history)


print("\nTraining Completed Successfully.")
print("\nBest model saved to:")
print(MODEL_DIR / "best_model.keras")