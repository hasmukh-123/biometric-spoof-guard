"""
Model A: Physical Presentation Attack Detection (PAD)
Architecture: Dual-Branch EfficientNet-B0 + Local Binary Pattern (LBP) Texture CNN.
Detects printed photos, screen replay attacks, and physical presentation artifacts.
"""
import torch
import torch.nn as nn
import timm


class PhysicalPADNet(nn.Module):
    def __init__(self, pretrained_backbone: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.rgb_backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained_backbone,
            num_classes=0,
        )
        rgb_dim = self.rgb_backbone.num_features

        self.texture_branch = nn.Sequential(
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

    def forward(self, rgb: torch.Tensor, lbp: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.rgb_backbone(rgb)
        tex_feat = self.texture_branch(lbp)
        fused = torch.cat([rgb_feat, tex_feat], dim=1)
        return self.classifier(fused).squeeze(-1)
