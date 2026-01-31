"""
Multi-scale cross-level fusion modules for FMC-ULite model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCrossLevelFusion(nn.Module):
    """
    Multi-scale cross-level fusion module with attention mechanisms
    """

    def __init__(self, in_channels_list, out_channels, dropout_rate=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.num_branches = len(in_channels_list)
        self.dropout_rate = dropout_rate

        # Shared convolution kernels for multi-scale processing
        self.shared_conv1x1 = nn.Sequential(
            nn.Conv2d(max(in_channels_list), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.Dropout2d(p=dropout_rate),  # Add Spatial Dropout
            nn.ReLU()
        )

        self.shared_conv_dilated = nn.ModuleDict({
            'd2': nn.Sequential(
                nn.Conv2d(max(in_channels_list), out_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels, momentum=0.1),
                nn.Dropout2d(p=dropout_rate),  # Add Spatial Dropout
                nn.ReLU()
            ),
            'd4': nn.Sequential(
                nn.Conv2d(max(in_channels_list), out_channels, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm2d(out_channels, momentum=0.1),
                nn.Dropout2d(p=dropout_rate),  # Add Spatial Dropout
                nn.ReLU()
            )
        })

        # Adapter layers for each input feature (parameter reduction)
        self.adapter_convs = nn.ModuleList()
        for ch in in_channels_list:
            if ch != max(in_channels_list):
                self.adapter_convs.append(
                    nn.Sequential(
                        nn.Conv2d(ch, max(in_channels_list), kernel_size=1),
                        nn.BatchNorm2d(max(in_channels_list)),
                        nn.Dropout2d(p=dropout_rate)  # Add Dropout
                    )
                )
            else:
                self.adapter_convs.append(nn.Identity())

        # Simplified spatial pyramid pooling
        self.pyramid_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((64, 64)),
            nn.Conv2d(max(in_channels_list), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.Dropout2d(p=dropout_rate),  # Add Dropout
            nn.ReLU()
        )

        # Simplified multi-scale attention (with Dropout)
        self.scale_attention = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.Dropout2d(p=dropout_rate),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.Sigmoid()
        )

        # Simplified channel attention (using SE block, with Dropout)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 8, 1), kernel_size=1),
            nn.BatchNorm2d(max(out_channels // 8, 1)),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # Standard dropout for fully connected layers
            nn.Conv2d(max(out_channels // 8, 1), out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.Sigmoid()
        )

        # More efficient feature fusion (with aggressive Dropout)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * self.num_branches, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.Dropout2d(p=dropout_rate * 1.5),  # Stronger dropout for fusion layer
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU()
        )

        # Add output dropout layer
        self.output_dropout = nn.Dropout2d(p=dropout_rate)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Add weight decay equivalent to L2 regularization
                m.weight.data = m.weight.data * 0.99  # Slight weight decay
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # Use smaller momentum for training stability
                m.momentum = 0.1

    def forward(self, features):
        """
        Optimized forward pass with reduced computational complexity

        Args:
            features: List of input feature tensors

        Returns:
            Fused feature tensor
        """
        processed_features = []
        target_size = features[-1].shape[2:]  # Use last layer as reference

        # Apply dropout during training, disable during evaluation
        self.apply_dropout(self.training)

        for i, (adapter, feat) in enumerate(zip(self.adapter_convs, features)):
            # Adapt feature channels (apply dropout)
            adapted_feat = adapter(feat)

            # Multi-scale feature extraction
            scale_features = []

            # Scale 1: 1x1 convolution
            scale1 = self.shared_conv1x1(adapted_feat)
            if scale1.shape[2:] != target_size:
                scale1 = F.interpolate(scale1, size=target_size, mode='bilinear', align_corners=True)
            scale_features.append(scale1)

            # Scale 2: Dilated convolution
            scale2 = self.shared_conv_dilated['d2'](adapted_feat)
            if scale2.shape[2:] != target_size:
                scale2 = F.interpolate(scale2, size=target_size, mode='bilinear', align_corners=True)
            scale_features.append(scale2)

            # Scale 3: Larger dilated convolution
            scale3 = self.shared_conv_dilated['d4'](adapted_feat)
            if scale3.shape[2:] != target_size:
                scale3 = F.interpolate(scale3, size=target_size, mode='bilinear', align_corners=True)
            scale_features.append(scale3)

            # Scale 4: Spatial pyramid pooling
            pooled = self.pyramid_pool(adapted_feat)
            scale4 = F.interpolate(pooled, size=target_size, mode='bilinear', align_corners=True)
            scale_features.append(scale4)

            # Concatenate multi-scale features
            multi_scale_feat = torch.cat(scale_features, dim=1)

            # Multi-scale attention (apply dropout)
            attention_weights = self.scale_attention(multi_scale_feat)
            attention_weights = attention_weights.view(-1, 4, self.out_channels, 1, 1)

            # Apply attention weights
            weighted_sum = torch.zeros_like(scale_features[0])
            for j in range(4):
                feat_slice = multi_scale_feat[:, j * self.out_channels:(j + 1) * self.out_channels]
                weight = attention_weights[:, j]
                weighted_sum = weighted_sum + feat_slice * weight

            # Channel attention (apply dropout)
            channel_weights = self.channel_attention(weighted_sum)
            final_feat = weighted_sum * channel_weights

            # Add feature dropout
            if self.training and self.dropout_rate > 0:
                final_feat = F.dropout2d(final_feat, p=self.dropout_rate / 2, training=True)

            processed_features.append(final_feat)

        # Fuse all features
        if len(processed_features) > 1:
            # Adjust sizes to match
            for i in range(len(processed_features) - 1):
                if processed_features[i].shape[2:] != processed_features[-1].shape[2:]:
                    processed_features[i] = F.interpolate(
                        processed_features[i],
                        size=processed_features[-1].shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )

            # Concatenate and fuse
            all_features = torch.cat(processed_features, dim=1)
        else:
            all_features = processed_features[0]

        fused_output = self.fusion_conv(all_features)

        # Final output dropout
        fused_output = self.output_dropout(fused_output)

        return fused_output

    def apply_dropout(self, mode=True):
        """Control the state of all dropout layers"""
        for module in self.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.train(mode)


class UpBlock(nn.Module):
    """
    Upsampling block with attention gate for decoder path
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU()
        )

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU()
        )

        # Attention gate
        from model.attention import DisasterAttentionGate
        self.attention_gate = DisasterAttentionGate(out_channels)

    def forward(self, x, skip):
        """
        Forward pass of upsampling block

        Args:
            x: Input tensor from decoder
            skip: Skip connection tensor from encoder

        Returns:
            Fused feature tensor
        """
        x = self.up(x)

        # Adjust skip connection size
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Attention mechanism
        attention = self.attention_gate(x)
        skip = skip * attention

        # Fuse features
        x = torch.cat([x, skip], dim=1)
        return self.fusion(x)