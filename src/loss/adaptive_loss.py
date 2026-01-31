"""
Adaptive combined loss function for segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveCombinedLoss(nn.Module):
    """
    Adaptive loss combining CrossEntropy and Dice loss
    """

    def __init__(self, class_weights, max_epochs):
        super().__init__()
        self.class_weights = class_weights
        self.max_epochs = max_epochs
        self.ce = nn.CrossEntropyLoss(weight=class_weights)  # Standard cross-entropy

    def forward(self, pred, target, current_epoch):
        """
        Calculate adaptive combined loss

        Args:
            pred: Prediction tensor [B, C, H, W] or [B*H*W, C]
            target: Target tensor [B, H, W] or [B*H*W]
            current_epoch: Current training epoch

        Returns:
            Combined loss value
        """
        target = target.long()

        # Ensure correct input dimensions
        if pred.dim() == 4 and target.dim() == 3:
            # Convert prediction from [B, C, H, W] to [B*H*W, C]
            # Convert target from [B, H, W] to [B*H*W]
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
            target = target.view(-1)

        # Calculate adaptive weight
        alpha = 0.5 * (1 + math.cos(math.pi * current_epoch / self.max_epochs))

        # Cross-entropy loss
        ce_loss = self.ce(pred, target)

        # Dice loss (maintain original implementation)
        pred_softmax = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.size(1)).float()
        intersection = torch.sum(pred_softmax * target_onehot, dim=0)
        union = torch.sum(pred_softmax + target_onehot, dim=0)
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        return alpha * ce_loss + (1 - alpha) * dice_loss