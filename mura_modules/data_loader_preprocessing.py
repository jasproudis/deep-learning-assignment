import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.applications import efficientnet, resnet, inception_v3
from config import BACKBONE_IMAGE_SIZES, DEFAULT_BACKBONE

# === Preprocessing ===
def get_preprocessing_fn(backbone):
    if backbone == "EfficientNetB0":
        return efficientnet.preprocess_input
    elif backbone == "ResNet50":
        return resnet.preprocess_input
    elif backbone == "InceptionV3":
        return inception_v3.preprocess_input
    else:
        raise ValueError("Unsupported backbone.")
    

'''

# === Label encoding for body parts ===
BODY_PART_LOOKUP = {
    'XR_ELBOW': 0,
    'XR_FINGER': 1,
    'XR_FOREARM': 2,
    'XR_HAND': 3,
    'XR_HUMERUS': 4,
    'XR_SHOULDER': 5,
    'XR_WRIST': 6,
    'OTHER': 7
}

def decode_hybrid_row(image_path, label_bin, body_part, weight):
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_raw, channels=1)               # Always grayscale → shape (H, W, 1)
    img = tf.image.resize(img, decode_hybrid_row.image_size)
    img = tf.image.grayscale_to_rgb(img)                      # Converts to shape (H, W, 3)
    img = tf.cast(img, tf.float32)
    img = decode_hybrid_row.preprocessing_fn(img)

    label_bin = tf.cast(label_bin, tf.float32)
    weight = tf.cast(weight, tf.float32)
    body_part = tf.cast(body_part, tf.int32)

    return img, body_part, label_bin, weight



def wrap_decode_hybrid_row(image_size, preprocessing_fn, part_to_index_map):
    decode_hybrid_row.image_size = image_size
    decode_hybrid_row.preprocessing_fn = preprocessing_fn
    decode_hybrid_row.part_to_index_map = part_to_index_map

    def wrapper(image_path, label_bin, body_part, weight):
        img, body_part_index, label_bin_out, weight_out = tf.py_function(
            func=decode_hybrid_row,
            inp=[image_path, label_bin, body_part, weight],
            Tout=[tf.float32, tf.int32, tf.float32, tf.float32]
        )
        img.set_shape(image_size + (3,))
        body_part_index.set_shape(())
        label_bin_out.set_shape(())
        weight_out.set_shape(())
        return (img, body_part_index, label_bin_out, weight_out)

    return wrapper

def wrap_decode_hybrid_row_validation(image_size, preprocessing_fn, part_to_index_map):
    decode_hybrid_row.image_size = image_size
    decode_hybrid_row.preprocessing_fn = preprocessing_fn
    decode_hybrid_row.part_to_index_map = part_to_index_map

    def wrapper(image_path, label_bin, body_part, weight):
        img, body_part_index, label_bin_out, weight_out = tf.py_function(
            func=decode_hybrid_row,
            inp=[image_path, label_bin, body_part, weight],
            Tout=[tf.float32, tf.int32, tf.float32, tf.float32]
        )
        img.set_shape(image_size + (3,))
        body_part_index.set_shape(())
        label_bin_out.set_shape(())
        weight_out.set_shape(())
        return (img, body_part_index, label_bin_out, weight_out)

    return wrapper


'''
def decode_hybrid_row(image_path, label_bin, body_part, weight):
    img_raw = tf.io.read_file(image_path)
    img = tf.io.decode_png(img_raw, channels=1)
    img = tf.image.resize(img, decode_hybrid_row.image_size)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.cast(img, tf.float32)
    img = decode_hybrid_row.preprocessing_fn(img)

    label_bin = tf.cast(label_bin, tf.float32)
    weight = tf.cast(weight, tf.float32)
    body_part = tf.cast(body_part, tf.int32)

    # One-hot encode body part as input
    body_part_one_hot = tf.one_hot(body_part, depth=decode_hybrid_row.num_categories + 1)

    return img, body_part_one_hot, label_bin, weight


def wrap_decode_hybrid_row_onehot(image_size, preprocessing_fn, part_to_index_map):
    decode_hybrid_row.image_size = image_size
    decode_hybrid_row.preprocessing_fn = preprocessing_fn
    decode_hybrid_row.part_to_index_map = part_to_index_map
    decode_hybrid_row.num_categories = 8

    def wrapper(image_path, label_bin, body_part, weight):
        img, body_part_one_hot, label_bin_out, weight_out = tf.py_function(
            func=decode_hybrid_row,
            inp=[image_path, label_bin, body_part, weight],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32]
        )
        img.set_shape(image_size + (3,))
        body_part_one_hot.set_shape((decode_hybrid_row.num_categories + 1,))
        label_bin_out.set_shape(())
        weight_out.set_shape(())

        return {'input_image': img, 'input_body_part_onehot': body_part_one_hot}, label_bin_out #, weight_out

    return wrapper




# === Dataset builder ===
def get_mura_dataset_hybrid(csv_path, image_size=None, batch_size=32, training=True, cache_dir=None, backbone=DEFAULT_BACKBONE):
    df = pd.read_csv(csv_path)
    image_paths = df['image_path'].astype(str).tolist()
    binary_labels = df['label'].astype(np.int32).tolist()
    body_parts = df['bodypart'].astype(str).tolist()

    # 🔧 Create part-to-index map
    unique_parts = sorted(set(body_parts))
    part_to_index_map = {part: idx for idx, part in enumerate(unique_parts)}
    body_part_indices = [part_to_index_map[part] for part in body_parts]

    # ✅ Add sample weights
    sample_weights = compute_sample_weight(class_weight='balanced', y=binary_labels)

    # ✅ Build TF dataset
    ds = tf.data.Dataset.from_tensor_slices((image_paths, binary_labels, body_part_indices, sample_weights))

    preprocessing_fn = get_preprocessing_fn(backbone)
    image_size = BACKBONE_IMAGE_SIZES[backbone] if image_size is None else image_size

    # ✅ Choose the correct wrapper for training or validation
    if training:
        map_fn = wrap_decode_hybrid_row_onehot(image_size, preprocessing_fn, part_to_index_map)
    else:
        map_fn = wrap_decode_hybrid_row_onehot(image_size, preprocessing_fn, part_to_index_map)

    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(1000)

    if cache_dir:
        ds = ds.cache(cache_dir + "/cache.tfdata")
    else:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
