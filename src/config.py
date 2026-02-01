"""
Configuration file for FMC-ULite model
Contains hyperparameters, paths, and model settings
"""

import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(42)

# Hyperparameters
IN_CHANNELS = 3
OUT_CHANNELS = 11  # Number of classes
BATCH_SIZE = 8
NUM_EPOCHS = 600
LEARNING_RATE = 0.001
PATIENCE = 50

# Paths
CHECKPOINT_PATH = "FMC-ULite_checkpoint.pth"
BEST_MODEL_PATH = "FMC-ULite_best_model.pth"
FINAL_MODEL_PATH = "FMC-ULite_final_model.pth"
EXCEL_METRICS_PATH = "training_metrics_FMC-ULite.xlsx"

# Dataset paths
TRAIN_IMAGE_DIR = "RescueNet/trainset/images"
TRAIN_MASK_DIR = "RescueNet/trainset/masks"
VAL_IMAGE_DIR = "RescueNet/validationset/images"
VAL_MASK_DIR = "RescueNet/validationset/masks"
TEST_IMAGE_DIR = "RescueNet/testset/images"
TEST_MASK_DIR = "RescueNet/testset/masks"

# Output directories
COLORMASK_TRAIN_DIR = "RescueNet/FMC-ULite/colormasks/train"
COLORMASK_VAL_DIR = "RescueNet/FMC-ULite/colormasks/validation"
COLORMASK_TEST_DIR = "RescueNet/FMC-ULite/colormasks/test"

# Class weights for loss function (RescueNet dataset)
CLASS_WEIGHTS = torch.tensor([
    0.0089,  # class 0: unlabeled
    0.0573,  # class 1: water
    0.1759,  # class 2: building-no-damage
    0.1770,  # class 3: building-medium-damage
    0.2770,  # class 4: building-major-damage
    0.3231,  # class 5: building-total-destruction
    1.4054,  # class 6: vehicle
    0.0672,  # class 7: road-clear
    0.2935,  # class 8: road-blocked
    0.0211,  # class 9: tree
    8.1935   # class 10: pool
])

# Model parameters
DROPOUT_RATE = 0.1
IMAGE_SIZE = (512, 512)

# Data augmentation parameters
FLIP_PROBABILITY = 0.3
COLOR_JITTER_PARAMS = {
    'brightness': 0.3,
    'contrast': 0.3,
    'saturation': 0.3,
    'hue': 0.1
}

RANDOM_GRAYSCALE_PROB = 0.1

# Normalization parameters (RescueNet dataset statistics)
NORMALIZE_MEAN = [0.5258, 0.5168, 0.4771]
NORMALIZE_STD = [0.2401, 0.2334, 0.2320]

