"""
Validation module for FMC-ULite model
"""

import torch
from tqdm import tqdm

from data.dataset import remove_num_classes_dim
from utils.metrics import mean_iou


def validate(model, dataloader, criterion, device, start_epoch):
    """
    Validate the model

    Args:
        model: The neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        start_epoch: Current epoch number

    Returns:
        val_loss: Average validation loss
        val_class_iou: Per-class IoU scores
        val_mean_iou: Mean IoU across all classes
    """
    model.eval()
    running_loss = 0.0
    running_class_iou = torch.zeros(11).to(device)
    running_count = 0

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc=f"Validating", unit="batch"):
            images = images.to(device)
            masks = masks.to(device).long()
            masks = remove_num_classes_dim(masks)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks, start_epoch)

            # Calculate predictions and IoU
            preds = torch.argmax(outputs, dim=1)
            class_iou, _ = mean_iou(preds, masks, 11)

            # Accumulate metrics
            running_loss += loss.item()
            running_class_iou += class_iou
            running_count += 1

    # Calculate average metrics
    val_loss = running_loss / len(dataloader)
    val_class_iou = running_class_iou / running_count
    val_mean_iou = torch.mean(val_class_iou).item()

    return val_loss, val_class_iou, val_mean_iou