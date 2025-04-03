import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
import tensorflow as tf

from cnn_model import build_multitask_cnn

# === Split data ===
X_train, X_val, y_train_bin, y_val_bin, y_train_cat, y_val_cat = train_test_split(
    images, labels_binary, labels_bodypart, test_size=0.2, random_state=42, stratify=labels_bodypart
)

# === Convert binary labels to float (for sigmoid) ===
y_train_bin = y_train_bin.astype(np.float32)
y_val_bin = y_val_bin.astype(np.float32)

# === One-hot encode body parts ===
y_train_cat = to_categorical(y_train_cat)
y_val_cat = to_categorical(y_val_cat)

# === Load model ===
model = build_multitask_cnn(input_shape=X_train.shape[1:], num_body_parts=y_train_cat.shape[1])

# === Compile ===
model.compile(
    optimizer='adam',
    loss={
        'binary_output': 'binary_crossentropy',
        'bodypart_output': 'categorical_crossentropy'
    },
    metrics={
        'binary_output': ['accuracy', Precision(), Recall()],
        'bodypart_output': 'accuracy'
    }
)

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=3, factor=0.2, monitor='val_loss'),
    ModelCheckpoint('best_multitask_model.h5', save_best_only=True, monitor='val_loss')
]

# === Train ===
history = model.fit(
    X_train, {'binary_output': y_train_bin, 'bodypart_output': y_train_cat},
    validation_data=(X_val, {'binary_output': y_val_bin, 'bodypart_output': y_val_cat}),
    batch_size=64,
    epochs=50,
    callbacks=callbacks
)

# === Evaluate on val set ===
pred_bin = (model.predict(X_val)[0] > 0.5).astype(int)
f1 = f1_score(y_val_bin, pred_bin)
print(f"Validation F1 Score (Binary Output): {f1:.4f}")
