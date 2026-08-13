"""
---------------------------------------------------------
VERINOTE

Module : MobileNetV2 Model

Purpose:
Builds the Deep Learning model for
counterfeit currency detection.

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

    # Freeze pretrained layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))

    x = data_augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model