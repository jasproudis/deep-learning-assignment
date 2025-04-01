# config.py
IMAGE_SIZE = (244,244)
ALLOWED_CLASSES = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']
OTHER_CLASS_NAME = 'OTHER'
BATCH_SIZE = 64

# Add these new ones for metrics/evaluation
NUM_CLASSES_BINARY = 1          # Binary classification (normal/abnormal)
NUM_CLASSES_BODYPART = len(ALLOWED_CLASSES) + 1  # +1 for 'OTHER' class