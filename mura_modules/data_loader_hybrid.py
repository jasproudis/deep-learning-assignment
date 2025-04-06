# data_loader_hybrid.py
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.utils.class_weight import compute_sample_weight

# Define mapping from body part to integer ID (include 'UNKNOWN')
BODY_PARTS = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM', 'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']
BODY_PART_TO_INDEX = {part: i for i, part in enumerate(BODY_PARTS)}
BODY_PART_TO_INDEX['UNKNOWN'] = len(BODY_PARTS)


# Decode and preprocess image, return with body part index and sample weight
def decode_hybrid_row(image_path, label_bin, body_part_str, weight):
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_raw, channels=1)
    img = tf.image.resize(img, (244, 244))
    img = tf.image.grayscale_to_rgb(img)
    img = tf.cast(img, tf.float32) / 255.0

    # Map string to index or 'UNKNOWN'
    part_index = BODY_PART_TO_INDEX.get(body_part_str.numpy().decode('utf-8'), BODY_PART_TO_INDEX['UNKNOWN'])
    part_index = tf.convert_to_tensor(part_index, dtype=tf.int32)

    return (img, part_index), label_bin, weight


def decode_hybrid_wrapper(image_path, label_bin, body_part_str, weight):
    return tf.py_function(func=decode_hybrid_row, inp=[image_path, label_bin, body_part_str, weight], Tout=((tf.float32, tf.int32), tf.int32, tf.float32))


def get_mura_dataset_hybrid(csv_path, image_size=(244, 244), batch_size=32, training=True, cache_dir=None):
    df = pd.read_csv(csv_path)
    df['body_part'] = df['image_path'].apply(lambda x: x.split('/')[-3].upper())

    image_paths = df['image_path'].astype(str).tolist()
    labels = df['label'].astype(np.int32).tolist()
    parts = df['body_part'].astype(str).tolist()

    weights = compute_sample_weight(class_weight='balanced', y=labels)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels, parts, weights))
    ds = ds.map(decode_hybrid_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)

    if cache_dir:
        cache_path = os.path.join(cache_dir, 'mura_hybrid_cache.tf-data')
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    ds = ds.map(lambda x, y, w: (x, y, w))  # Remove the outer label+weight nesting
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
