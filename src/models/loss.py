import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., 2017)
    Paper: https://arxiv.org/abs/1708.02002
    
    Reduces the loss contribution from easy examples and increases the importance of hard examples.
    Formula: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        """
        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class.
                                     If given, has to be a Tensor of size `C`.
            gamma (float, optional): Focusing parameter. Default: 2.0
            reduction (string, optional): Specifies the reduction to apply to the output:
                                        'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [Batch_Size, Num_Classes] (Logits)
            targets: [Batch_Size] (Class Indices)
        """
        # 1. Compute Cross Entropy Loss (log_pt) without reduction
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # 2. Get the probabilities (pt) for the correct classes
        pt = torch.exp(-ce_loss)
        
        # 3. Calculate Focal Component: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Apply Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
