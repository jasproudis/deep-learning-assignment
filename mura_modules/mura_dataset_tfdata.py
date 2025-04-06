# mura_dataset.py

import tensorflow as tf
import pandas as pd
import numpy as np
from config import ALLOWED_CLASSES, OTHER_CLASS_NAME, IMAGE_SIZE, BATCH_SIZE
from sklearn.utils.class_weight import compute_sample_weight
import os

def parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    body_parts = sorted(set(df['bodypart'].tolist()) | {OTHER_CLASS_NAME})
    bodypart_to_index = {bp: i for i, bp in enumerate(body_parts)}
    return df, bodypart_to_index

def decode_image(image_path, image_size):
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_raw, channels=1)
    img = tf.image.resize(img, image_size)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def decode_binary_row(image_path, label_bin, weight):
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_raw, channels=1)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.grayscale_to_rgb(img)  # convert to 3 channels
    img = tf.cast(img, tf.float32) / 255.0
    return img, label_bin, weight

def process_row(image_path, label_bin, label_part, image_size):
    image = decode_image(image_path, image_size)
    return image, {
        'binary_output': label_bin,
        'bodypart_output': label_part
    }

def augment_image(image, label, weight=None):
    # Apply random transformations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    if weight is not None:
        return image, label, weight
    else:
        return image, label

def get_mura_dataset(csv_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=True, cache_dir=None):
    df, bodypart_to_index = parse_csv(csv_path)

    image_paths = df['image_path'].astype(str).tolist()
    binary_labels = df['label'].astype(np.int32).tolist()
    bodypart_labels = [bodypart_to_index[bp] for bp in df['bodypart']]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, binary_labels, bodypart_labels))
    ds = ds.map(lambda path, bin_lbl, part_lbl: process_row(path, bin_lbl, part_lbl, image_size),
                num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(buffer_size=1000)

        if cache_dir:
            cache_path = f"{cache_dir}/mura_train_cache.tf-data"
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_mura_dataset_binary_only(csv_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=True, cache_dir=None):


    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].astype(str).tolist()
    binary_labels = df['label'].astype(np.int32).tolist()

    # Compute sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=binary_labels)

    # Create the dataset
    ds = tf.data.Dataset.from_tensor_slices((image_paths, binary_labels, sample_weights))


    ds = ds.map(decode_binary_row, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)

    if cache_dir:
        cache_path = os.path.join(cache_dir, "mura_train_cache.tf-data")
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def get_mura_dataset_binary_only_augmented(csv_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, training=True, cache_dir=None):


    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].astype(str).tolist()
    binary_labels = df['label'].astype(np.int32).tolist()

    # Compute sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=binary_labels)

    # Create the dataset
    ds = tf.data.Dataset.from_tensor_slices((image_paths, binary_labels, sample_weights))


    ds = ds.map(decode_binary_row, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_dir:
        cache_path = os.path.join(cache_dir, "mura_train_cache.tf-data")
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
