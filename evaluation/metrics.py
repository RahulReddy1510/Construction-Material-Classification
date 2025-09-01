import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def compute_accuracy(preds, labels):
    """Simple top-1 accuracy."""
    preds_idx = np.argmax(preds, axis=1)
    return accuracy_score(labels, preds_idx)

def compute_per_class_accuracy(preds, labels, class_names):
    """Computes accuracy for each material class."""
    preds_idx = np.argmax(preds, axis=1)
    cm = confusion_matrix(labels, preds_idx, labels=range(len(class_names)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return {name: f"{acc*100:.2f}%" for name, acc in zip(class_names, per_class_acc)}

def get_full_report(preds, labels, class_names):
    """Generates a text classification report."""
    preds_idx = np.argmax(preds, axis=1)
    return classification_report(labels, preds_idx, target_names=class_names)

def compute_confusion_matrix(preds, labels, class_names):
    """Returns a raw confusion matrix."""
    preds_idx = np.argmax(preds, axis=1)
    return confusion_matrix(labels, preds_idx)

if __name__ == "__main__":
    # Demo verification
    classes = ["concrete", "brick", "metal", "wood", "stone"]
    y_true = [0, 1, 2, 3, 4, 0, 1, 2]
    y_pred_logits = np.random.randn(8, 5) # random garbage
    
    print("--- Metrics Test ---")
    print(f"Top-1 Accuracy: {compute_accuracy(y_pred_logits, y_true):.4f}")
    print(f"Per-Class Accuracy: {compute_per_class_accuracy(y_pred_logits, y_true, classes)}")
