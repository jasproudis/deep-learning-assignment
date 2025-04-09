# create_image_metadata_csv.py

import os
import glob
import pandas as pd
from tqdm import tqdm
from config import (
    BASE_PATH_PC,
    BASE_PATH_GOOGLE_DRIVE,
    ALLOWED_CLASSES,
    OTHER_CLASS_NAME
)

def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def get_base_path():
    return BASE_PATH_GOOGLE_DRIVE if in_colab() else BASE_PATH_PC

def extract_image_metadata(csv_filename, output_csv_filename, base_path):
    """
    Create an expanded CSV with image-level labels.
    """
    df = pd.read_csv(csv_filename, names=['study', 'label'], skiprows=1)

    image_paths, labels, body_parts = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(csv_filename)}"):
        study_path = row['study']
        label = int(row['label'])
        class_name = next((part.upper() for part in study_path.split('/') if part.upper().startswith("XR_")), OTHER_CLASS_NAME)
        
        stripped_class = class_name.replace("XR_", "")
        if stripped_class not in ALLOWED_CLASSES:
            print(f"[WARN] Unknown class_name: {class_name}")
            class_name = OTHER_CLASS_NAME
        else:
            class_name = stripped_class



        study_dir = os.path.join(base_path, study_path.replace("MURA-v1.1/", "").replace("/", os.sep))
        pngs = glob.glob(os.path.join(study_dir, "*.png"))

        for img_path in pngs:
            img_path = img_path.replace(base_path, BASE_PATH_GOOGLE_DRIVE).replace("\\", "/")
            image_paths.append(img_path)
            labels.append(label)
            body_parts.append(class_name)

    out_df = pd.DataFrame({
        "image_path": image_paths,
        "label": labels,
        "bodypart": body_parts
    })

    out_df.to_csv(output_csv_filename, index=False)
    print(f"✅ Saved to: {output_csv_filename}")

if __name__ == "__main__":
    base_path = get_base_path()
    
    train_csv = os.path.join(base_path, "train_labeled_studies.csv")
    val_csv = os.path.join(base_path, "valid_labeled_studies.csv")

    out_train = os.path.join(base_path, "train_image_metadata.csv")
    out_val = os.path.join(base_path, "valid_image_metadata.csv")

    extract_image_metadata(train_csv, out_train, base_path)
    extract_image_metadata(val_csv, out_val, base_path)
