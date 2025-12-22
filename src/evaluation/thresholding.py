
import numpy as np
from sklearn.metrics import f1_score

def optimize_thresholds(y_true, y_probs, num_labels=6):
    """
    Finds the optimal threshold for each label that maximizes F1 score.
    Search range: 0.1 to 0.9.
    """
    best_thresholds = np.full(num_labels, 0.5)
    best_f1s = np.zeros(num_labels)
    
    threshold_range = np.arange(0.1, 0.95, 0.05)
    
    for label_idx in range(num_labels):
        y_true_label = y_true[:, label_idx]
        y_prob_label = y_probs[:, label_idx]
        
        best_f1 = 0
        best_thresh = 0.5
        
        for thresh in threshold_range:
            y_pred = (y_prob_label >= thresh).astype(int)
            score = f1_score(y_true_label, y_pred, zero_division=0)
            
            if score > best_f1:
                best_f1 = score
                best_thresh = thresh
                
        best_thresholds[label_idx] = best_thresh
        best_f1s[label_idx] = best_f1
        
    return best_thresholds, best_f1s
