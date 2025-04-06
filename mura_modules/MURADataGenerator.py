# data_generator.py

import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Sequence
from config import IMAGE_SIZE , BATCH_SIZE

class MURADataGenerator(Sequence):
    def __init__(self, csv_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df['image_path'].tolist()
        self.binary_labels = self.df['label'].tolist()
        self.bodypart_labels = self.df['bodypart'].tolist()
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Encode bodypart labels
        self.bodypart_encoder = {label: idx for idx, label in enumerate(sorted(set(self.bodypart_labels)))}
        self.encoded_bodyparts = [self.bodypart_encoder[bp] for bp in self.bodypart_labels]
        
        # self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        # batch_indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]        
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels_bin = self.binary_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels_part = self.encoded_bodyparts[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_images = np.zeros((len(batch_paths), *self.image_size, 3), dtype=np.float32)

        for i, path in enumerate(batch_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Warning] Could not read image: {path}")
                continue
            img = cv2.resize(img, self.image_size)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            batch_images[i] = img.astype(np.float32) / 255.0


        return batch_images, {
            "binary_output": np.array(batch_labels_bin),
            "bodypart_output": np.array(batch_labels_part)
        }

    def on_epoch_end(self):
        if self.shuffle:
            zipped = list(zip(self.image_paths, self.binary_labels, self.bodypart_labels))
            np.random.shuffle(zipped)
            self.image_paths, self.binary_labels, self.bodypart_labels = zip(*zipped)
            self.encoded_bodyparts = [self.bodypart_encoder[bp] for bp in self.bodypart_labels]
