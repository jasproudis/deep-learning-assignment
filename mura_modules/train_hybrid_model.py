# train_hybrid_model.py
import tensorflow as tf
from hybrid_model import build_hybrid_model
from data_loader_hybrid import get_mura_dataset_hybrid
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import datetime

# === Settings ===
CSV_TRAIN = "MURA_train_labeled.csv"
CSV_VALID = "MURA_valid_labeled.csv"
BATCH_SIZE = 32
EPOCHS = 10

# === Load datasets ===
train_ds = get_mura_dataset_hybrid(CSV_TRAIN, batch_size=BATCH_SIZE, training=True)
valid_ds = get_mura_dataset_hybrid(CSV_VALID, batch_size=BATCH_SIZE, training=False)

# === Build model ===
model = build_hybrid_model(image_shape=(244, 244, 3), num_categories=8, embedding_dim=16)
model.summary()

# === Callbacks ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
checkpoint_cb = ModelCheckpoint(
    filepath=f"hybrid_model_best_{timestamp}.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

earlystop_cb = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
lr_reduce_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)

# === Train ===
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, lr_reduce_cb],
    verbose=1
)

# === Optional: Save final model ===
model.save(f"hybrid_model_final_{timestamp}.h5")
