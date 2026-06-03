import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs fake images.
    Uses pretrained VGG16 as the convolutional backbone (features[:18]),
    which corresponds to conv1_1 through conv3_3 plus one extra conv
    (256→512), ending at 512×64×64 for a 512×512 input.
    The backbone is frozen by default.

    Input : (batch, 3, 512, 512)   — 3-channel, ImageNet-normalised
    Output: (batch, 1) logits      — positive = real, negative = fake
    Use with BCEWithLogitsLoss for training.

    VGG16 features slice reference (512×512 input):
        [:18]  → 512ch × 64×64   (through conv3_3 + one 256→512 conv)
        [:24]  → 512ch × 32×32   (through pool4)
    The AdaptiveAvgPool2d(4,4) makes spatial size irrelevant for the FC head.
    """

    def __init__(
        self,
        input_channels=3,
        use_spectral_norm: bool = True,
        freeze_backbone: bool = True,
        vgg_layers: int = 18,
    ):
        super(Discriminator, self).__init__()

        self.use_spectral_norm = use_spectral_norm

        # ── Pretrained VGG16 backbone ────────────────────────────────────────
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(vgg.features.children())[:vgg_layers])

        # features[:18] ends at Conv2d(256→512), so output is 512 channels.
        backbone_out_channels = 512

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Helper ───────────────────────────────────────────────────────────
        def sn(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # ── Extra conv on top of backbone ────────────────────────────────────
        # Reduces spatial size by 2 more (64→32 for 512×512 input).
        self.extra_conv = nn.Sequential(
            sn(nn.Conv2d(backbone_out_channels, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Classifier head ──────────────────────────────────────────────────
        # AdaptiveAvgPool2d(4,4) makes this input-size agnostic.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1      = sn(nn.Linear(512 * 4 * 4, 1024))
        self.dropout1 = nn.Dropout(0.5)

        self.fc2      = sn(nn.Linear(1024, 256))
        self.dropout2 = nn.Dropout(0.5)

        self.fc_out   = sn(nn.Linear(256, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W)  — normalise with ImageNet stats before passing:
               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std =[0.229, 0.224, 0.225])

        Returns:
            logits: (batch, 1)
        """
        x = self.backbone(x)        # (B, 512, 64, 64)  for 512×512 input
        x = self.extra_conv(x)      # (B, 512, 32, 32)

        x = self.adaptive_pool(x)   # (B, 512, 4, 4)
        x = x.view(x.size(0), -1)  # (B, 8192)

        x = self.leaky_relu(self.fc1(x))
        x = self.dropout1(x)

        x = self.leaky_relu(self.fc2(x))
        x = self.dropout2(x)

        return self.fc_out(x)       # (B, 1) — raw logits