"""
Training module for FMC-ULite model
"""

import torch
from tqdm import tqdm
import torch.nn.functional as F

from data.dataset import remove_num_classes_dim
from utils.metrics import mean_iou


def train(model, dataloader, criterion, optimizer, device, accumulation_steps, start_epoch):
    """
    Train the model for one epoch

    Args:
        model: The neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        accumulation_steps: Gradient accumulation steps
        start_epoch: Current epoch number

    Returns:
        epoch_loss: Average training loss
        epoch_class_iou: Per-class IoU scores
        epoch_mean_iou: Mean IoU across all classes
    """
    model = model.float()
    model.train()
    running_loss = 0.0
    running_class_iou = torch.zeros(11).to(device)  # Store sum of IoU for each class
    running_count = 0

    pbar = tqdm(dataloader, desc="Training", unit="batch",
                postfix={'loss': 'NaN', 'mIoU': 'NaN'})

    for i, (images, masks, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True, dtype=torch.float32)
        masks = masks.to(device, non_blocking=True).long()
        masks = remove_num_classes_dim(masks)

        # Forward pass
        outputs = model(images)

        # Check for NaN values
        if torch.isnan(outputs).any():
            print("NaN in model outputs! Skipping batch...")
            continue

        # Calculate loss
        loss = criterion(outputs, masks, start_epoch)

        # Backward pass with gradient accumulation
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        running_loss += loss.item()

        # Update weights based on accumulation steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Calculate IoU
        preds = torch.argmax(outputs, dim=1)
        class_iou, _ = mean_iou(preds, masks, 11)
        running_class_iou += class_iou
        running_count += 1  # Increment counter

        # Update progress bar dynamically
        pbar.set_postfix({
            'loss': f"{running_loss / (i + 1):.4f}",
            'mIoU': f"{torch.mean(running_class_iou / (i + 1)).item():.4f}",
        }, refresh=True)

    # Calculate epoch average metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_class_iou = running_class_iou / running_count  # Average IoU per class
    epoch_mean_iou = torch.mean(epoch_class_iou).item()  # Average mIoU

    return epoch_loss, epoch_class_iou, epoch_mean_iou