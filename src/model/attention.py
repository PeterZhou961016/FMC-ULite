"""
Attention mechanism modules for FMC-ULite model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DisasterAttentionGate(nn.Module):
    """
    Disaster attention gate with multi-scale dilated convolutions
    """

    def __init__(self, channels):
        super().__init__()
        # Multi-scale dilated convolutions
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels // 4, 3, dilation=d, padding=d)
            for d in [1, 3, 6]  # Multi-scale dilation rates
        ])
        self.fusion = nn.Conv2d(channels // 4 * 3, channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the disaster attention gate

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Attention map [B, C, H, W]
        """
        # Generate attention maps from different dilation rates
        attn_maps = [conv(x) for conv in self.dilated_convs]

        # Fuse attention maps and apply sigmoid
        attn = torch.sigmoid(self.fusion(torch.cat(attn_maps, dim=1)))

        return attn