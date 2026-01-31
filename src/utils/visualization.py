"""
Visualization utilities for segmentation results
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import torch


def create_colormap(num_classes):
    """
    Create color map for segmentation visualization

    Args:
        num_classes: Number of classes

    Returns:
        List of RGB colors for each class
    """
    # Define default colors (can be extended as needed)
    colors = [
        [0, 0, 0],  # Class 0: unlabeled
        [61, 230, 250],  # Class 1: water
        [180, 120, 120],  # Class 2: building-no-damage
        [235, 255, 7],  # Class 3: building-medium-damage
        [255, 184, 6],  # Class 4: building-major-damage
        [255, 0, 0],  # Class 5: building-total-destruction
        [255, 0, 245],  # Class 6: vehicle
        [140, 140, 140],  # Class 7: road-clear
        [160, 150, 20],  # Class 8: road-blocked
        [4, 250, 7],  # Class 9: tree
        [255, 235, 0]  # Class 10: pool
    ]

    # If more classes than predefined colors, generate random colors
    if num_classes > len(colors):
        import random
        for _ in range(num_classes - len(colors)):
            colors.append([random.randint(0, 255) for _ in range(3)])

    return colors[:num_classes]


def convert_to_colormap(pred_mask, num_classes):
    """
    Convert prediction mask to color image

    Args:
        pred_mask: Prediction mask [H, W] with class indices
        num_classes: Number of classes

    Returns:
        Color mask image [H, W, 3]
    """
    colormap = create_colormap(num_classes)

    # Initialize color image
    height, width = pred_mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Map class indices to colors
    for i in range(num_classes):
        color_mask[pred_mask == i] = colormap[i]

    return color_mask


def save_colormasks(model, dataloader, device, save_dir, num_classes):
    """
    Save color-coded segmentation masks

    Args:
        model: Trained model
        dataloader: Data loader for images
        device: Device to run inference on
        save_dir: Directory to save color masks
        num_classes: Number of classes
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(dataloader)):
        images, masks, image_files = batch  # Extract images, masks and filenames
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).long()
            preds = torch.clamp(preds, 0, num_classes - 1)

        # Save prediction results
        for j in range(preds.shape[0]):
            # Get original filename (without extension)
            original_filename = os.path.splitext(image_files[j])[0]

            # Generate color mask
            pred_mask = preds[j].cpu().numpy()
            color_mask_rgb = convert_to_colormap(pred_mask, num_classes)

            # Generate save path
            save_path = os.path.join(save_dir, f"{original_filename}_cmsk.png")

            # Save color mask (convert RGB to BGR for OpenCV)
            color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, color_mask_bgr)