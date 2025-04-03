# plot_utils.py

import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    if 'binary_output_accuracy' in history.history:
        plt.plot(history.history['binary_output_accuracy'], label='Train Binary Acc')
        plt.plot(history.history['val_binary_output_accuracy'], label='Val Binary Acc')
    if 'bodypart_output_accuracy' in history.history:
        plt.plot(history.history['bodypart_output_accuracy'], label='Train BodyPart Acc')
        plt.plot(history.history['val_bodypart_output_accuracy'], label='Val BodyPart Acc')
    plt.title("Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
