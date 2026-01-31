"""
Data transformation and augmentation functions
"""

import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as Fu


class SyncRandomFlip:
    """
    Synchronized random flipping for image-mask pairs
    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = Fu.hflip(img)
            mask = Fu.hflip(mask)
        if random.random() < self.p:
            img = Fu.vflip(img)
            mask = Fu.vflip(mask)
        return img, mask


def get_train_transforms():
    """
    Get training data transformations with augmentation

    Returns:
        Composition of training transforms for images and masks
    """
    from config import (
        IMAGE_SIZE, COLOR_JITTER_PARAMS, RANDOM_GRAYSCALE_PROB,
        GAUSSIAN_BLUR_PARAMS, NORMALIZE_MEAN, NORMALIZE_STD
    )

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ColorJitter(**COLOR_JITTER_PARAMS),
        transforms.RandomGrayscale(p=RANDOM_GRAYSCALE_PROB),
        transforms.GaussianBlur(**GAUSSIAN_BLUR_PARAMS),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    mask_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    ])

    return image_transform, mask_transform


def get_val_test_transforms():
    """
    Get validation/test data transformations (no augmentation)

    Returns:
        Composition of validation/test transforms for images and masks
    """
    from config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

    mask_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    ])

    return image_transform, mask_transform