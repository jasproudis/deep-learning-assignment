# hybrid_model.py
# Hybrid model with image + embedded category input, supporting EfficientNetB0, ResNet50, and InceptionV3

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, InceptionV3

def build_hybrid_model(
    image_shape=(224, 224, 3),
    num_categories=8,
    backbone="ResNet50",  # or "InceptionV3"
    dropout_rate=0.3,
    dense_units=128,
    base_trainable=False
):
    # Input 1: Image
    image_input = tf.keras.Input(shape=image_shape, name='input_image')

    if backbone == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    elif backbone == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=image_input)
    else:
        raise ValueError("Unsupported backbone. Use 'ResNet50' or 'InceptionV3'")

    base_model.trainable = base_trainable
    x = layers.GlobalAveragePooling2D()(base_model.output)

    # Input 2: One-hot body part vector
    bodypart_input = tf.keras.Input(shape=(num_categories + 1,), name='input_body_part_onehot')

    # Combine both towers
    combined = layers.Concatenate()([x, bodypart_input])
    combined = layers.Dense(dense_units, activation='relu')(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    output = layers.Dense(1, activation='sigmoid', name='binary_output')(combined)

    model = tf.keras.Model(inputs=[image_input, bodypart_input], outputs=output)
    model._name = f"hybrid_{backbone.lower()}_onehot"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model, base_model

def fine_tune_hybrid_model(model, base_model, unfreeze_from=100, learning_rate=1e-5):
    """
    Unfreezes layers in the base model starting from `unfreeze_from` and recompiles the full hybrid model.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model