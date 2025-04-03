# metrics.py
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np

def get_binary_metrics(y_true, y_pred):
    """
    Calculates F1 Score, Accuracy, Precision, Recall for binary classification
    Args:
        y_true (np.array): Ground truth binary labels
        y_pred (np.array): Predicted binary labels (logits or probabilities)
    Returns:
        dict: dictionary with metric names and values
    """
    # Convert logits/probs to 0/1 if needed
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
    y_pred_bin = (y_pred >= 0.5).astype(int)

    return {
        "f1": f1_score(y_true, y_pred_bin),
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "precision": precision_score(y_true, y_pred_bin),
        "recall": recall_score(y_true, y_pred_bin)
    }
