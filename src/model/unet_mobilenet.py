"""
Main U-Net architecture with MobileNetV3 backbone for FMC-ULite
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from model.fft_fusion import FFTFusion
from model.fusion_modules import MultiScaleCrossLevelFusion, UpBlock


class UNetLiteWithMobileNetV3(nn.Module):
    """
    U-Net Lite architecture with MobileNetV3 backbone and FFT preprocessing
    """

    def __init__(self, n_classes=11):
        super().__init__()
        # Enhanced FFT preprocessing
        self.fft_preprocess = FFTFusion(in_channels=3)

        # Load MobileNetV3-large as backbone
        backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # Modify first convolution layer to accept 3-channel input (already processed by EnhancedFFTFusion)
        original_first_layer = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            3,  # Already processed by EnhancedFFTFusion
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False
        )

        # Encoder part
        self.encoder = nn.Sequential(
            backbone.features[0],  # /2 (256x256)
            backbone.features[1],  # /1
            backbone.features[2],  # /2 (128x128)
            backbone.features[3],  # /1
            backbone.features[4],  # /2 (64x64)
            backbone.features[5],  # /1
            backbone.features[6],  # /1
            backbone.features[7],  # /2 (32x32)
            backbone.features[8],  # /1
            backbone.features[9],  # /1
            backbone.features[10],  # /1
            backbone.features[11],  # /1
        )
        self.encoder_channels = [16, 16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112]

        # Cross-level feature fusion
        self.cross_level_fusion = MultiScaleCrossLevelFusion(
            in_channels_list=[16, 24, 40, 80, 112],
            out_channels=64
        )

        # Decoder part
        self.up1 = UpBlock(112, 80)  # 32x32 -> 64x64
        self.up2 = UpBlock(80, 40)  # 64x64 -> 128x128
        self.up3 = UpBlock(40, 24)  # 128x128 -> 256x256
        self.up4 = UpBlock(24, 16)  # 256x256 -> 512x512

        # Final output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(80, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(16, n_classes, kernel_size=1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the complete network

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Segmentation map [B, n_classes, H, W]
        """
        # FFT preprocessing
        x = self.fft_preprocess(x)

        # Encoder path (store features from each layer)
        features = []
        e0 = self.encoder[0](x)  # /2
        features.append(e0)
        e1 = self.encoder[1](e0)  # /1
        features.append(e1)
        e2 = self.encoder[2](e1)  # /2
        features.append(e2)
        e3 = self.encoder[3](e2)  # /1
        features.append(e3)
        e4 = self.encoder[4](e3)  # /2
        features.append(e4)
        e5 = self.encoder[5](e4)  # /1
        e6 = self.encoder[6](e5)  # /1
        e7 = self.encoder[7](e6)  # /2
        e8 = self.encoder[8](e7)  # /1
        e9 = self.encoder[9](e8)  # /1
        e10 = self.encoder[10](e9)  # /1
        e11 = self.encoder[11](e10)  # /1 (32x32)

        # Cross-level feature fusion
        selected_features = [features[0], features[2], features[4], e7, e11]
        fused_features = self.cross_level_fusion(selected_features)

        # Decoder path
        d1 = self.up1(e11, e9)  # 64x64
        d2 = self.up2(d1, e5)  # 128x128
        d3 = self.up3(d2, e3)  # 256x256
        d4 = self.up4(d3, e1)  # 512x512

        # Fuse cross-level features
        if d4.shape[2:] != fused_features.shape[2:]:
            fused_features = F.interpolate(fused_features, size=d4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, fused_features], dim=1)

        # Final classification
        out = self.out_conv(d4)
        return out