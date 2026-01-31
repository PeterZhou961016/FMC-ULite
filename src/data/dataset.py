"""
Dataset classes for RescueNet dataset
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data.transforms import SyncRandomFlip


class RescueNetDataset(Dataset):
    """
    RescueNet dataset loader with synchronized transformations
    """

    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.augment = augment
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.num_classes = 11
        self.sync_flip = SyncRandomFlip(p=0.3)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace(".jpg", "_lab.png"))

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # Data augmentation
        if self.augment:
            if torch.rand(1) < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1) < 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        # Apply transformations separately
        if self.image_transform:
            image = self.image_transform(image)
            # Validate input range
            assert image.min() >= -3 and image.max() <= 3, \
                f"Input out of range: [{image.min():.3f}, {image.max():.3f}]"

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Clamp mask values to valid class range
        mask = torch.clamp(mask, 0, self.num_classes - 1)

        return image, mask, self.image_files[idx]


def remove_num_classes_dim(masks):
    """
    Remove unnecessary dimension from mask tensors

    Args:
        masks: Input mask tensor [B, 1, H, W] or [B, C, H, W]

    Returns:
        Squeezed mask tensor [B, H, W]
    """
    return masks[:, 0, :, :]