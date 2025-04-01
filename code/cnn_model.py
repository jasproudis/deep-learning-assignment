from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def build_cnn_model(hp, input_shape=(224, 224, 1), num_classes_binary=1, num_classes_bodypart=8):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Tune the number of conv blocks
    for i in range(hp.Int('conv_blocks', 2, 4)):
        filters = hp.Choice(f'filters_{i}', values=[32, 64, 128])
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(hp.Float('dropout', 0.3, 0.6, step=0.1))(x)
    x = layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu')(x)

    # Multitask output branches
    output_binary = layers.Dense(num_classes_binary, activation='sigmoid', name='binary_output')(x)
    output_bodypart = layers.Dense(num_classes_bodypart, activation='softmax', name='bodypart_output')(x)

    model = Model(inputs=inputs, outputs=[output_binary, output_bodypart])

    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
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
