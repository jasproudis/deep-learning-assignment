import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall

from cnn_model import build_multitask_cnn

# === Split data ===
X_train, X_val, y_train_bin, y_val_bin, y_train_cat, y_val_cat = train_test_split(
    images, labels_binary, labels_bodypart, test_size=0.2, random_state=42, stratify=labels_bodypart
)

y_train_bin = y_train_bin.astype(np.float32)
y_val_bin = y_val_bin.astype(np.float32)
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

# === TensorBoard log directory ===
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=3, factor=0.2, monitor='val_loss'),
    ModelCheckpoint('best_multitask_model.h5', save_best_only=True, monitor='val_loss'),
    tensorboard_cb  # comment this out if not needed
]

# === Train ===
history = model.fit(
    X_train, {'binary_output': y_train_bin, 'bodypart_output': y_train_cat},
    validation_data=(X_val, {'binary_output': y_val_bin, 'bodypart_output': y_val_cat}),
    batch_size=64,
    epochs=50,
    callbacks=callbacks
)

# === Evaluate and print F1, Precision, Recall for binary output ===
pred_bin = (model.predict(X_val)[0] > 0.5).astype(int)
f1 = f1_score(y_val_bin, pred_bin)
precision = precision_score(y_val_bin, pred_bin)
recall = recall_score(y_val_bin, pred_bin)
print(f"\nF1 Score: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

# === Plot training curves ===
def plot_history(history):
    # Binary classification accuracy
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_output_accuracy'], label='Train Binary Acc')
    plt.plot(history.history['val_binary_output_accuracy'], label='Val Binary Acc')
    plt.plot(history.history['bodypart_output_accuracy'], label='Train BodyPart Acc')
    plt.plot(history.history['val_bodypart_output_accuracy'], label='Val BodyPart Acc')
    plt.title("Accuracy Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)
