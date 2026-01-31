"""
Checkpoint management utilities for saving and loading model states
"""

import os
import torch
import pandas as pd


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Dictionary with loaded state information
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Previous best validation IoU: {checkpoint.get('best_iou', 0):.4f}")

    return checkpoint


def save_checkpoint(checkpoint_path, epoch, model, optimizer=None,
                    best_iou=0.0, train_losses=None, val_losses=None, val_ious=None):
    """
    Save model checkpoint

    Args:
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save (optional)
        best_iou: Best IoU value achieved
        train_losses: List of training losses
        val_losses: List of validation losses
        val_ious: List of validation IoUs
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_iou': best_iou,
        'train_losses': train_losses or [],
        'val_losses': val_losses or [],
        'val_ious': val_ious or [],
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")


def load_training_metrics(excel_path):
    """
    Load training metrics from Excel file

    Args:
        excel_path: Path to Excel file

    Returns:
        DataFrame with training metrics
    """
    if os.path.exists(excel_path):
        metrics_df = pd.read_excel(excel_path)
        print(f"Loaded existing training records, total {len(metrics_df)} epochs")
    else:
        metrics_df = pd.DataFrame(columns=[
            'Epoch',
            'Train Loss',
            'Train mIoU',
            'Val Loss',
            'Val mIoU',
            'Learning Rate'
        ])
        print(f"Created new metrics DataFrame")

    return metrics_df


def save_training_metrics(metrics_df, excel_path):
    """
    Save training metrics to Excel file

    Args:
        metrics_df: DataFrame with metrics
        excel_path: Path to save Excel file

    Returns:
        Success flag
    """
    try:
        metrics_df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Training metrics saved to {excel_path}")
        return True
    except Exception as e:
        print(f"Failed to save Excel file: {str(e)}")
        return False