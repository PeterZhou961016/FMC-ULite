"""
Testing module for FMC-ULite model
"""

import time
import torch
from tqdm import tqdm

from data.dataset import remove_num_classes_dim
from utils.metrics import mean_iou
from fvcore.nn import FlopCountAnalysis


def test(model, dataloader, device):
    """
    Test the model on test dataset

    Args:
        model: The neural network model
        dataloader: Test data loader
        device: Device to run testing on

    Returns:
        test_class_iou: Per-class IoU scores
        test_mean_iou: Mean IoU across all classes
        fps: Frames per second (inference speed)
    """
    model.eval()
    total_class_iou = torch.zeros(11).to(device)  # Store sum of IoU for each class
    total_count = 0  # Count samples
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc=f"Testing", unit="batch"):
            images = images.to(device)
            masks = masks.to(device).long()
            masks = remove_num_classes_dim(masks)

            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            torch.cuda.synchronize()
            total_time += time.time() - start_time

            # Calculate predictions and IoU
            preds = torch.argmax(outputs, dim=1)
            class_iou, _ = mean_iou(preds, masks, 11)

            # Accumulate metrics
            total_class_iou += class_iou.to(device)
            total_count += 1
            total_samples += images.size(0)

    # Calculate average metrics
    avg_class_iou = total_class_iou / total_count  # Average IoU per class
    avg_mean_iou = torch.mean(avg_class_iou)  # Average mIoU
    fps = total_samples / total_time

    return avg_class_iou, avg_mean_iou, fps


def calculate_flops(model, device, input_size=(1, 3, 512, 512)):
    """
    Calculate FLOPs (Floating Point Operations) of the model

    Args:
        model: The neural network model
        device: Device to run calculation on
        input_size: Input tensor size for FLOP calculation

    Returns:
        Total FLOPs count
    """
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    return flops.total()