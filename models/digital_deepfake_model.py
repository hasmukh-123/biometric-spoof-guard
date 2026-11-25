"""
Model B: Digital Deepfake & Face Manipulation Detection
Architecture: Dual-Stream Xception + 2D FFT Frequency Artifact Branch.
Detects face-swaps, GAN-generated faces, and deepfake video manipulations.
"""
import torch
import torch.nn as nn
import timm


class DigitalDeepfakeNet(nn.Module):
    def __init__(self, pretrained_backbone: bool = True, dropout: float = 0.35) -> None:
        super().__init__()
        self.rgb_backbone = timm.create_model(
            "xception",
            pretrained=pretrained_backbone,
            num_classes=0,
        )
        rgb_dim = self.rgb_backbone.num_features

        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, rgb: torch.Tensor, fft_map: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(rgb)
        freq_feat = self.freq_branch(fft_map)
        fused = torch.cat([rgb_feat, freq_feat], dim=1)
        return self.classifier(fused).squeeze(-1)
