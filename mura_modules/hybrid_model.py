# hybrid_model.py
# Hybrid model with image + embedded category input, supporting EfficientNetB0, ResNet50, and InceptionV3

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, InceptionV3

def build_hybrid_model(image_shape=(244, 244, 3), num_categories=8, embedding_dim=16, backbone="EfficientNetB0"):
    # Input 1: Image
    image_input = tf.keras.Input(shape=image_shape, name='input_image')

    # Backbone selection
    if backbone == "EfficientNetB0":
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=image_input)
    elif backbone == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    elif backbone == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=image_input)
    else:
        raise ValueError("Unsupported backbone. Choose from 'EfficientNetB0', 'ResNet50', or 'InceptionV3'")

    base_model.trainable = False  # Freeze for now
    x = layers.GlobalMaxPooling2D()(base_model.output)

    # Input 2: Body part category (int)
    category_input = tf.keras.Input(shape=(1,), dtype='int32', name='input_body_part')
    embedding = layers.Embedding(input_dim=num_categories + 1, output_dim=embedding_dim)(category_input)
    embedding = layers.Flatten()(embedding)

    # Concatenate image features and body part embedding
    combined = layers.Concatenate()([x, embedding])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.3)(combined)
    output = layers.Dense(1, activation='sigmoid', name='binary_output')(combined)

    model = tf.keras.Model(inputs=[image_input, category_input], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model
