# model_pretrained_binary.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_binary_pretrained_model(input_shape, num_classes_binary=1, backbone="EfficientNetB0"):
    if backbone == "EfficientNetB0":
        base_model = EfficientNetB0(include_top=False,
                                    weights="imagenet",
                                    input_shape=input_shape,
                                    pooling="avg")
    else:
        raise ValueError("Unsupported backbone.")

    base_model.trainable = False  # Initially freeze base model

    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    output_binary = layers.Dense(num_classes_binary, activation="sigmoid", name="binary_output")(x)

    model = models.Model(inputs=inputs, outputs=output_binary)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, base_model


def fine_tune_binary_model(model, base_model, unfreeze_from=150, learning_rate=1e-5):
    """
    Unfreezes layers from `unfreeze_from` onward in base_model for fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
