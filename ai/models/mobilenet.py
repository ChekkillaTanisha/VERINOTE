"""
---------------------------------------------------------
VERINOTE

Module : MobileNetV2 Model

Purpose:
Builds the MobileNetV2-based binary classification model.

Author : Tanisha Chekkilla
---------------------------------------------------------
"""

import tensorflow as tf

from ai.data.augmentation import data_augmentation
from ai.config.training import IMAGE_SIZE, DROPOUT_RATE


def build_model():

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers except the last 30 layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Input
    inputs = tf.keras.Input(
        shape=IMAGE_SIZE + (3,)
    )

    # Data augmentation
    x = data_augmentation(inputs)

    # MobileNetV2 preprocessing
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Feature extraction
    x = base_model(
        x,
        training=False
    )

    # Global feature representation
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Regularization
    x = tf.keras.layers.Dropout(
        DROPOUT_RATE
    )(x)

    # Two-class output:
    # 0 = Fake
    # 1 = Real
    outputs = tf.keras.layers.Dense(
        2,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs,
        outputs
    )

    return model