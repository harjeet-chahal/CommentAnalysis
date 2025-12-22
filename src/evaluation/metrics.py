
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score
import numpy as np

def calculate_metrics(y_true, y_pred, y_probs=None, average='macro'):
    """
    Calculates F1, Precision, Recall.
    If y_probs is provided, also calculates ROC-AUC and PR-AUC.
    """
    metrics = {
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs, average=average)
            metrics['pr_auc'] = average_precision_score(y_true, y_probs, average=average)
        except ValueError:
            # Handle cases where a class might not be present in the batch/split
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
            
    return metrics
