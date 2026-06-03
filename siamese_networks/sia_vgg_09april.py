import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity comparison.
    Uses a shared pretrained VGG16 backbone (features[:18]) for feature
    extraction, frozen by default.

    Input : two images of (batch, 3, 512, 512) — ImageNet-normalised
    Output: (batch, 1) probability — 0 = similar, 1 = dissimilar

    VGG16 features slice reference (512×512 input):
        [:18]  → 512ch × 64×64   (features[:18] ends at Conv2d(256→512))
        [:24]  → 512ch × 32×32   (through pool4)
    The AdaptiveAvgPool2d(4,4) makes spatial size irrelevant for the FC head.
    """

    def __init__(
        self,
        input_channels=3,
        freeze_backbone: bool = True,
        vgg_layers: int = 18,
    ):
        super(SiameseNetwork, self).__init__()

        # ── Shared pretrained VGG16 backbone ─────────────────────────────────
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:vgg_layers])

        # features[:18] ends at Conv2d(256→512), so output is 512 channels.
        backbone_out_channels = 512

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Adaptive pooling ─────────────────────────────────────────────────
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Embedding head (applied identically to each branch) ───────────────
        self.fc1      = nn.Linear(backbone_out_channels * 4 * 4, 1024)  # 512*16=8192
        self.dropout1 = nn.Dropout(0.5)
        self.fc2      = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3      = nn.Linear(512, 256)

        # ── Similarity classifier ─────────────────────────────────────────────
        self.fc_out = nn.Linear(256, 1)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one branch: backbone → adaptive pool → embedding head.

        Args:
            x: (batch, 3, H, W) — normalised with ImageNet stats.
        Returns:
            embedding: (batch, 256)
        """
        x = self.backbone(x)            # (B, 512, 64, 64)  for 512×512 input
        x = self.adaptive_pool(x)       # (B, 512, 4, 4)
        x = x.view(x.size(0), -1)      # (B, 8192)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))         # (B, 256)
        return x

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2: (batch, 3, 512, 512) — normalise with ImageNet stats:
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std =[0.229, 0.224, 0.225])

        Returns:
            probability: (batch, 1) — 0 = similar, 1 = dissimilar
        """
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)

        difference = torch.abs(embedding1 - embedding2)

        return torch.sigmoid(self.fc_out(difference))