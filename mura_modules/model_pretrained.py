# model_pretrained.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

def build_pretrained_model(input_shape, num_classes_binary, num_classes_bodypart, backbone='efficientnetb0'):
    """
    Builds a multitask classification model using a pretrained backbone (EfficientNetB0 or MobileNetV2).
    """
    inputs = tf.keras.Input(shape=input_shape, name='input_image')

    if backbone == 'efficientnetb0':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
    elif backbone == 'mobilenetv2':
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    base_model.trainable = False  # Freeze base initially

    x = base_model.output
    x = layers.Dropout(0.3)(x)

    # Output 1: Binary classification (sigmoid), cast to float32
    output_binary = layers.Dense(
        num_classes_binary,
        activation='sigmoid',
        dtype='float32',               # Ensure numerical stability
        name='binary_output'
    )(x)

    # Output 2: Body part classification (softmax), cast to float32
    output_bodypart = layers.Dense(
        num_classes_bodypart,
        activation='softmax',
        dtype='float32',
        name='bodypart_output'
    )(x)

    model = models.Model(inputs=inputs, outputs=[output_binary, output_bodypart])

    return model
