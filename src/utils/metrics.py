"""
Metrics calculation utilities for segmentation tasks
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def mean_iou(pred, target, num_classes):
    """
    Calculate mean Intersection over Union (IoU)

    Args:
        pred: Prediction tensor [B, H, W] or [B, C, H, W]
        target: Target tensor [B, H, W]
        num_classes: Number of classes

    Returns:
        iou_tensor: Per-class IoU scores as tensor
        mean_iou_value: Mean IoU across all classes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resize if necessary
    if pred.dim() == 4:  # [B, C, H, W] -> need to argmax first
        pred = torch.argmax(pred, dim=1)

    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred.unsqueeze(1).float(),
                             size=target.shape[-2:],
                             mode='bilinear',
                             align_corners=True).squeeze(1).long()

    # Flatten tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Mask valid pixels
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]

    # Calculate confusion matrix
    cm = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy(),
                          labels=list(range(num_classes)))

    # Calculate IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
    iou = intersection / (union + 1e-6)

    return torch.tensor(iou, dtype=torch.float32, device=device), np.nanmean(iou)


def calculate_iou(pred, target, num_classes):
    """
    Alternative IoU calculation using PyTorch operations

    Args:
        pred: Prediction logits [B, C, H, W]
        target: Target tensor [B, H, W]
        num_classes: Number of classes

    Returns:
        Per-class IoU scores
    """
    # Convert predictions to class labels
    pred = torch.argmax(pred, dim=1)  # Assume pred is logits [B, C, H, W]

    # Initialize confusion matrix
    confusion_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    # Calculate confusion matrix for each sample
    for p, t in zip(pred, target):
        confusion_mat += torch.bincount(
            t.flatten() * num_classes + p.flatten(),
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

    # Calculate intersection (diagonal elements)
    intersection = torch.diag(confusion_mat)

    # Calculate union
    union = (
            confusion_mat.sum(dim=0) +  # Column sum (FP + TP)
            confusion_mat.sum(dim=1) -  # Row sum (FN + TP)
            intersection  # Subtract double-counted TP
    )

    # Calculate IoU for each class
    iou = intersection / union

    return iou