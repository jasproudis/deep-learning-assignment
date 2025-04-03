import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

class MuraSequence(Sequence):
    def __init__(self, image_paths, labels_binary, labels_bodypart, batch_size=32, image_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.labels_binary = labels_binary
        self.labels_bodypart = labels_bodypart
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.image_paths))

        batch_paths = self.image_paths[start:end]
        batch_bin = self.labels_binary[start:end]
        batch_part = self.labels_bodypart[start:end]

        X = np.array([self._load_image(p) for p in batch_paths])
        y_bin = np.array(batch_bin)
        y_part = np.array(batch_part)

        return X, {"binary_output": y_bin, "bodypart_output": y_part}

    def on_epoch_end(self):
        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels_binary = [self.labels_binary[i] for i in indices]
            self.labels_bodypart = [self.labels_bodypart[i] for i in indices]

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img.astype("float32") / 255.0
