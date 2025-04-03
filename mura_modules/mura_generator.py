import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence

class MURAGenerator(Sequence):
    def __init__(self, image_paths, labels_binary, labels_bodypart, batch_size=32,
                 image_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels_binary = labels_binary
        self.labels_bodypart = labels_bodypart
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Get batch items
        batch_image_paths = [self.image_paths[i] for i in batch_indices]
        batch_binary_labels = [self.labels_binary[i] for i in batch_indices]
        batch_bodypart_labels = [self.labels_bodypart[i] for i in batch_indices]

        # Load and preprocess images
        X = np.array([self._load_image(p) for p in batch_image_paths], dtype=np.float16)
        y_bin = np.array(batch_binary_labels, dtype=np.int32)
        y_part = np.array(batch_bodypart_labels, dtype=np.float32)

        return X, {"binary_output": y_bin, "bodypart_output": y_part}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # if model expects RGB
        img = img.astype("float16") / 255.0
        return img
