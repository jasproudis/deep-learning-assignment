# data_loader.py

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from config import IMAGE_SIZE, ALLOWED_CLASSES, OTHER_CLASS_NAME

def load_mura_dataset(base_path, csv_path):
    df = pd.read_csv(csv_path, names=['study', 'label'], skiprows=1)
    
    image_paths, labels, body_parts = [], [], []
    for _, row in df.iterrows():
        # Get class name (e.g. WRIST, ELBOW)
        study_path = row['study']
        class_name = study_path.split('/')[2].upper()
        label = int(row['label'])

        # Assign OTHER if not in ALLOWED_CLASSES
        if class_name not in ALLOWED_CLASSES:
            class_name = OTHER_CLASS_NAME

        study_dir = os.path.join(base_path, study_path.replace("MURA-v1.1/", "").replace("/", os.sep))
        for png in glob.glob(os.path.join(study_dir, "*.png")):
            image_paths.append(png)
            labels.append(label)
            body_parts.append(class_name)

    return image_paths, labels, body_parts


def preprocess_images(image_paths, image_size=IMAGE_SIZE):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L').resize(image_size)
        img = np.array(img).astype(np.float32) / 255.0
        images.append(np.expand_dims(img, axis=-1))  # Shape: (H, W, 1)
    return np.array(images)


def encode_labels(labels, body_parts):
    le_body_part = LabelEncoder()
    body_parts_encoded = le_body_part.fit_transform(body_parts)
    body_parts_cat = to_categorical(body_parts_encoded)

    return np.array(labels), body_parts_cat, le_body_part
