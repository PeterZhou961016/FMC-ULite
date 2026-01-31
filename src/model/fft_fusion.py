"""
FFT (Fast Fourier Transform) fusion module for frequency domain processing
"""

import torch
import torch.nn as nn
import torch.fft


class GaussianLowPassFilter(nn.Module):
    """
    Gaussian low-pass filter with learnable parameters for FFT processing
    """

    def __init__(self, init_sigma=30, init_alpha=0.1):
        super().__init__()
        # Learnable parameters
        self.sigma = nn.Parameter(torch.tensor(float(init_sigma)))  # Learnable sigma
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))  # Leakage coefficient

    def forward(self, x):
        """
        Apply Gaussian low-pass filter in frequency domain

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Concatenated tensor of original and filtered images [B, 2*C, H, W]
        """
        # Process each channel separately
        filtered_channels = []

        for c in range(x.size(1)):
            channel = x[:, c:c + 1, :, :]  # Get single channel [B, 1, H, W]

            # Compute FFT
            fft = torch.fft.fft2(channel, dim=(-2, -1))
            fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

            # Create Gaussian low-pass filter
            b, _, h, w = channel.shape
            center_y, center_x = h // 2, w // 2
            y, x_coord = torch.meshgrid(
                torch.arange(h, device=x.device),
                torch.arange(w, device=x.device),
                indexing='ij'
            )

            # Gaussian mask
            mask = torch.exp(-((y - center_y) ** 2 + (x_coord - center_x) ** 2) / (2 * self.sigma ** 2))
            mask = mask.view(1, 1, h, w)

            # Apply filter
            filtered_fft = fft_shifted * mask

            # Inverse FFT
            filtered_fft = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
            filtered = torch.fft.ifft2(filtered_fft).real

            filtered_channels.append(filtered)

        # Merge all channels
        filtered_image = torch.cat(filtered_channels, dim=1)

        # Concatenate original and filtered images
        return torch.cat([x, filtered_image], dim=1)


class FFTFusion(nn.Module):
    """
    FFT fusion module with spatial and channel attention mechanisms
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.fft_processor = GaussianLowPassFilter()

        # Calculate channel dimensions
        self.total_channels = in_channels * 2
        self.reduction_channels = max(1, self.total_channels // 8)  # Gentle compression

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(self.total_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # Channel attention (SE block variant)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.total_channels, self.reduction_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(self.reduction_channels),
            nn.ReLU(),
            nn.Conv2d(self.reduction_channels, self.total_channels,
                      kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.total_channels, in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of FFT fusion module

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Fused feature tensor [B, C, H, W]
        """
        # FFT processing
        fft_feat = self.fft_processor(x)

        # Spatial attention
        spatial_weights = self.spatial_attention(fft_feat)
        spatial_out = fft_feat * spatial_weights

        # Channel attention
        channel_weights = self.channel_attention(fft_feat)
        channel_out = fft_feat * channel_weights

        # Fusion
        fused = spatial_out + channel_out
        return self.fusion_conv(fused)