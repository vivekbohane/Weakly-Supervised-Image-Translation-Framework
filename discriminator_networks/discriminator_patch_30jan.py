# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Discriminator(nn.Module):
#     """
#     PatchGAN Discriminator for phase images.
#     Classifies 70x70 patches as real or fake.
#     Better for texture and local structure discrimination.
#     """
#     def __init__(self, in_channels=3, ndf=64):
#         """
#         Args:
#             in_channels: Number of input channels (3 for RGB)
#             ndf: Number of discriminator filters in first conv layer
#         """
#         super(Discriminator, self).__init__()
        
#         # 512x512 -> 256x256
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # 256x256 -> 128x128
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # 128x128 -> 64x64
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # 64x64 -> 32x32
#         self.layer4 = nn.Sequential(
#             nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True)
#         )
        
#         # 32x32 -> 31x31 (PatchGAN output)
#         self.layer5 = nn.Sequential(
#             nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
#             # No sigmoid here - will use BCEWithLogitsLoss
#         )
        
#     def forward(self, x):
#         """
#         Args:
#             x: Input image tensor [B, 3, 512, 512]
#         Returns:
#             Patch predictions [B, 1, 31, 31]
#         """
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x



import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super().__init__()

        def conv_block(in_c, out_c, stride, norm=True):
            layers = [
                spectral_norm(
                    nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)
                )
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []

        # First layer (no normalization)
        layers += conv_block(in_channels, base_channels, stride=2, norm=False)

        nf_mult = 1
        for i in range(1, n_layers):
            prev = base_channels * nf_mult
            nf_mult = min(2 ** i, 8)
            layers += conv_block(prev, base_channels * nf_mult, stride=2)

        # One stride-1 layer
        prev = base_channels * nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += conv_block(prev, base_channels * nf_mult, stride=1)

        # Output patch map
        layers.append(
            spectral_norm(
                nn.Conv2d(base_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        )

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        return self.model(x)
