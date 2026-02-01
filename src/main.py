"""
Main entry point for FMC-ULite model training and evaluation
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import configuration
import config as cfg

# Import modules
from model.unet_mobilenet import UNetLiteWithMobileNetV3
from data.dataset import RescueNetDataset
from data.transforms import get_train_transforms, get_val_test_transforms
from loss.adaptive_loss import AdaptiveCombinedLoss
from train.trainer import train
from train.validator import validate
from train.tester import test, calculate_flops
from utils.checkpoint import (
    load_checkpoint, save_checkpoint,
    load_training_metrics, save_training_metrics
)
from utils.visualization import save_colormasks


def create_data_loaders():
    """
    Create data loaders for training, validation, and testing

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_image_transform, train_mask_transform = get_train_transforms()
    val_image_transform, val_mask_transform = get_val_test_transforms()

    # Create datasets
    train_dataset = RescueNetDataset(
        image_dir=cfg.TRAIN_IMAGE_DIR,
        mask_dir=cfg.TRAIN_MASK_DIR,
        image_transform=train_image_transform,
        mask_transform=train_mask_transform,
    )

    val_dataset = RescueNetDataset(
        image_dir=cfg.VAL_IMAGE_DIR,
        mask_dir=cfg.VAL_MASK_DIR,
        image_transform=val_image_transform,
        mask_transform=val_mask_transform,
    )

    test_dataset = RescueNetDataset(
        image_dir=cfg.TEST_IMAGE_DIR,
        mask_dir=cfg.TEST_MASK_DIR,
        image_transform=val_image_transform,
        mask_transform=val_mask_transform,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=8
    )

    return train_loader, val_loader, test_loader


def setup_model_and_optimizer():
    """
    Initialize model, loss function, optimizer, and scheduler

    Returns:
        Tuple of (model, criterion, optimizer, scheduler)
    """
    # Initialize model
    model = UNetLiteWithMobileNetV3(n_classes=cfg.OUT_CHANNELS).to(cfg.device)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Initialize weights
    model._initialize_weights()

    # Move class weights to device
    class_weights = cfg.CLASS_WEIGHTS.to(cfg.device)

    return model, criterion, optimizer, scheduler


def main():
    """
    Main training and evaluation pipeline
    """
    print("Starting FMC-ULite training pipeline...")
    print("-" * 100)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders()

    # Setup model and optimizer
    print("Setting up model and optimizer...")
    model, criterion, optimizer, scheduler = setup_model_and_optimizer()

    # Load training metrics
    print("Loading training metrics...")
    metrics_df = load_training_metrics(cfg.EXCEL_METRICS_PATH)

   
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        current_epoch = epoch + 1
        print(f"Epoch {current_epoch}/{cfg.NUM_EPOCHS}:")

        # Training phase
        train_loss, train_class_iou, train_mean_iou = train(
            model, train_loader, criterion, optimizer,
            cfg.device, cfg.ACCUMULATION_STEPS, epoch
        )

        # Validation phase
        val_loss, val_class_iou, val_mean_iou = validate(
            model, val_loader, criterion, cfg.device, epoch
        )

        # Update training history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_mean_iou)

        # Print per-class IoU
        print(f"Train Class IoU: {train_class_iou}")
        print(f"Validation Class IoU: {val_class_iou}")

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"Epoch {current_epoch}/{cfg.NUM_EPOCHS}, "
              f"Learning Rate: {current_lr:.2e}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train mIoU: {train_mean_iou:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val IoU: {val_mean_iou:.4f}")

        # Update metrics DataFrame
        new_data = {
            'Epoch': [current_epoch],
            'Train Loss': [train_loss],
            'Train mIoU': [train_mean_iou],
            'Val Loss': [val_loss],
            'Val mIoU': [val_mean_iou],
            'Learning Rate': [current_lr]
        }

        if metrics_df.empty:
            metrics_df = pd.DataFrame(new_data)
        else:
            metrics_df = pd.concat([metrics_df, pd.DataFrame(new_data)], ignore_index=True)

        # Save metrics to Excel
        save_training_metrics(metrics_df, cfg.EXCEL_METRICS_PATH)

        # Save best model
        if val_mean_iou > best_iou:
            best_iou = val_mean_iou
            no_improve = 0
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"New best model saved with IoU: {best_iou:.4f}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")

        # Save checkpoint
        save_checkpoint(
            checkpoint_path, epoch, model, optimizer,
            best_iou, train_losses, val_losses, val_ious
        )

        print("-" * 100)

        # Early stopping check
        if no_improve >= cfg.PATIENCE:
            print(f"Early stopping at epoch {current_epoch}.")
            break

    # Load best model for testing
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH))

    # Testing phase
    print("Running test evaluation...")
    test_class_iou, test_mean_iou, fps = test(model, test_loader, cfg.device)

    # Calculate FLOPs
    flops = calculate_flops(model, cfg.device)

    # Print test results
    print("\n" + "=" * 100)
    print("TEST RESULTS:")
    print("=" * 100)
    print(f"Test Class IoU: {test_class_iou}")
    print(f"Test mIoU: {test_mean_iou.item():.4f}")
    print(f"Inference Speed: {fps:.2f} FPS")
    print(f"Computational Complexity: {flops / 1e9:.2f} GFLOPs")

    # Save color-coded segmentation masks
    print("\nSaving color-coded segmentation masks...")
    save_colormasks(
        model, train_loader, cfg.device,
        save_dir=cfg.COLORMASK_TRAIN_DIR,
        num_classes=cfg.OUT_CHANNELS
    )
    save_colormasks(
        model, val_loader, cfg.device,
        save_dir=cfg.COLORMASK_VAL_DIR,
        num_classes=cfg.OUT_CHANNELS
    )
    save_colormasks(
        model, test_loader, cfg.device,
        save_dir=cfg.COLORMASK_TEST_DIR,
        num_classes=cfg.OUT_CHANNELS
    )

    # Save final model
    print("Saving final model...")
    torch.save(model.state_dict(), cfg.FINAL_MODEL_PATH)

    print("\nTraining and evaluation completed successfully!")
    print(f"Best model saved at: {cfg.BEST_MODEL_PATH}")
    print(f"Final model saved at: {cfg.FINAL_MODEL_PATH}")
    print(f"Training metrics saved at: {cfg.EXCEL_METRICS_PATH}")


if __name__ == "__main__":
    main()
