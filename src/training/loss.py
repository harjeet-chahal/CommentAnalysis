
import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, class_counts, total_samples, device):
        """
        Args:
            class_counts (list or np.array): Number of positive samples for each class.
            total_samples (int): Total number of samples in the dataset.
            device (torch.device): Device to put the weights on.
        """
        super(WeightedBCELoss, self).__init__()
        
        # Calculate pos_weight for each class
        # pos_weight = number_of_negatives / number_of_positives
        pos_counts = torch.tensor(class_counts, dtype=torch.float)
        neg_counts = total_samples - pos_counts
        pos_weights = neg_counts / (pos_counts + 1e-6) # Add epsilon to avoid division by zero
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    def forward(self, logits, targets):
        return self.criterion(logits, targets)
