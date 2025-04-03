# mura_dataset.py

import numpy as np
import os

class MURADataset:
    def __init__(self, images, labels_binary, labels_bodypart, label_encoder):
        self.images = images
        self.labels_binary = labels_binary
        self.labels_bodypart = labels_bodypart
        self.label_encoder = label_encoder  # Optional: useful to decode body part names

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "images.npy"), self.images)
        np.save(os.path.join(folder, "labels_binary.npy"), self.labels_binary)
        np.save(os.path.join(folder, "labels_bodypart.npy"), self.labels_bodypart)
        np.save(os.path.join(folder, "label_classes.npy"), self.label_encoder.classes_)

    @classmethod
    def load(cls, folder):
        images = np.load(os.path.join(folder, "images.npy"))
        labels_binary = np.load(os.path.join(folder, "labels_binary.npy"))
        labels_bodypart = np.load(os.path.join(folder, "labels_bodypart.npy"))
        classes = np.load(os.path.join(folder, "label_classes.npy"))

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = classes

        return cls(images, labels_binary, labels_bodypart, le)
