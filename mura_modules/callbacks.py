# callbacks.py

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import os
import datetime

def get_callbacks(model_name='best_model.h5', use_tensorboard=False):
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_name,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    if use_tensorboard:
        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

    return callbacks
