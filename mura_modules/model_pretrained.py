import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def build_pretrained_model(input_shape, num_classes_binary, num_classes_bodypart):
    inputs = tf.keras.Input(shape=input_shape, name='input_image')

    # Base EfficientNetB0 with ImageNet weights
    base_model = EfficientNetB0(include_top=False,
                                 weights='imagenet',
                                 input_tensor=inputs,
                                 pooling='avg')  # global average pooling

    # Freeze base model initially
    base_model.trainable = False

    # Shared features
    x = base_model.output
    x = layers.Dropout(0.3)(x)  # Regularization

    # Output 1: Binary classification (normal/abnormal)
    output_binary = layers.Dense(num_classes_binary, activation='sigmoid', name='binary_output')(x)

    # Output 2: Body part classification
    output_bodypart = layers.Dense(num_classes_bodypart, activation='softmax', name='bodypart_output')(x)

    model = models.Model(inputs=inputs, outputs=[output_binary, output_bodypart])

    # Compile with multitask loss
    model.compile(
        optimizer='adam',
        loss={
            'binary_output': 'binary_crossentropy',
            'bodypart_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'binary_output': ['accuracy'],
            'bodypart_output': ['accuracy']
        }
    )

    return model

def fine_tune_model(model, base_model, unfreeze_from=100):
    """
    Unfreeze last `n` layers for fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss={
            'binary_output': 'binary_crossentropy',
            'bodypart_output': 'sparse_categorical_crossentropy'
        },
        metrics={
            'binary_output': ['accuracy'],
            'bodypart_output': ['accuracy']
        }
    )
    return model
