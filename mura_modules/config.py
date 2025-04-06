# config.py

# === Input Settings ===
DEFAULT_BACKBONE = "EfficientNetB0"

# Backbone-specific image sizes
BACKBONE_IMAGE_SIZES = {
    "EfficientNetB0": (244, 244),
    "ResNet50": (244, 244),
    "InceptionV3": (299, 299)
}

# Dynamically set based on backbone
IMAGE_SIZE = BACKBONE_IMAGE_SIZES[DEFAULT_BACKBONE]

# === Dataset Settings ===
ALLOWED_CLASSES = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']
OTHER_CLASS_NAME = 'OTHER'
BATCH_SIZE = 32

BASE_PATH_PC = 'C:\\Users\\jasproudis\\Desktop\\MscDataScience\\Deep Learning\\MURA-v1.1\\'
BASE_PATH_GOOGLE_DRIVE = '/content/drive/MyDrive/Colab Notebooks/Deep_Learning/MURA-v1.1/'

# Add these new ones for metrics/evaluation
NUM_CLASSES_BINARY = 1          # Binary classification (normal/abnormal)
NUM_CLASSES_BODYPART = len(ALLOWED_CLASSES) + 1  # +1 for 'OTHER' class
NUM_CATEGORIES_HYBRID = NUM_CLASSES_BODYPART

CSV_TRAIN = BASE_PATH_GOOGLE_DRIVE + 'MURA_train_labeled.csv'
CSV_VALID = BASE_PATH_GOOGLE_DRIVE + 'MURA_valid_labeled.csv'
CACHE_DIR = None
