import torch
from torch import nn
from torchvision import models

def conv3x(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)

class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """Pretrained U-Net generator based on TernausNet"""
    def __init__(self, num_filters=32, pretrained=True, in_channels=1, out_channels=3):
        """
        if pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = models.vgg11(pretrained=pretrained).features
        self.conv321 = ConvRelu(1, 3)
        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]
        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlock(num_filters * 8, num_filters * 8 * 2, num_filters * 4)  # Modified
        self.dec3 = DecoderBlock(num_filters * 4, num_filters * 4 * 2, num_filters * 2)  # Modified
        self.dec2 = DecoderBlock(num_filters * 2, num_filters * 2 * 2, num_filters)      # Modified
        self.dec1 = ConvRelu(num_filters, num_filters)                                    # Modified
        self.final = nn.Conv2d(num_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv321 = self.conv321(x)
        conv1 = self.relu(self.conv1(conv321))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(dec5)            # Modified
        dec3 = self.dec3(dec4)            # Modified
        dec2 = self.dec2(dec3)            # Modified
        dec1 = self.dec1(dec2)            # Modified
        dec0 = self.final(dec1)
        return self.sigmoid(dec0), None

