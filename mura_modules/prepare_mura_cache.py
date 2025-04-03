import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2

# ---- Config ----
IMAGE_SIZE = (244, 244)
ALLOWED_CLASSES = {'WRIST', 'ELBOW', 'SHOULDER', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS'}
OTHER_CLASS_NAME = 'OTHER'

def load_image_paths_and_labels(base_path, csv_path):
    df = pd.read_csv(csv_path, names=['study', 'label'], skiprows=1)

    image_paths, labels, body_parts = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading image paths from {csv_path}"):
        study_path = row['study']
        label = int(row['label'])
        class_name = study_path.split('/')[2].upper()

        if class_name not in ALLOWED_CLASSES:
            class_name = OTHER_CLASS_NAME

        study_dir = os.path.join(base_path, study_path.replace("MURA-v1.1/", "").replace("/", os.sep))
        for png in glob.glob(os.path.join(study_dir, "*.png")):
            image_paths.append(png)
            labels.append(label)
            body_parts.append(class_name)

    return image_paths, np.array(labels), np.array(body_parts)

def preprocess_images(image_paths, image_size=(244, 244)):
    images = []
    for path in tqdm(image_paths, desc="Preprocessing images"):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Expand to 3 channels
        img = img.astype("float16") / 255.0      # Use float16 here in order to load the model in RAM and save 50% without accuracy cost.
        images.append(img)
    return np.array(images, dtype=np.float16)

def encode_body_parts(body_parts):
    le = LabelEncoder()
    encoded = le.fit_transform(body_parts)
    return to_categorical(encoded), le

def save_npz(filename, images, labels_binary, labels_bodypart):
    np.savez_compressed(filename, images=images, labels_binary=labels_binary, labels_bodypart=labels_bodypart)

def prepare_and_save_dataset(base_path, csv_name, output_path):
    csv_path = os.path.join(base_path, csv_name)
    image_paths, labels_binary, body_parts = load_image_paths_and_labels(base_path, csv_path)
    images = preprocess_images(image_paths)
    labels_bodypart, _ = encode_body_parts(body_parts)
    save_npz(output_path, images, labels_binary, labels_bodypart)

# ---- Main ----
if __name__ == "__main__":
    BASE_PATH = "C:\\Users\\jasproudis\\Desktop\\MscDataScience\\Deep Learning\\MURA-v1.1"
    CACHE_DIR = "C:\\Users\\jasproudis\\Desktop\\MscDataScience\\Deep Learning"
    os.makedirs(CACHE_DIR, exist_ok=True)

    prepare_and_save_dataset(BASE_PATH, "train_labeled_studies.csv", f"{CACHE_DIR}/mura_train_data.npz")
    prepare_and_save_dataset(BASE_PATH, "valid_labeled_studies.csv", f"{CACHE_DIR}/mura_valid_data.npz")

    print("✅ Done! Saved")
