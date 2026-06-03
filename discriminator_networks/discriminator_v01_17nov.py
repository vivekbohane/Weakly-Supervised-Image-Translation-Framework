import torch
import torch.nn as nn
import torch.nn.functional as F

# Add option to select input channels if needed in future

class Discriminator(nn.Module):
    """
    Discriminator network for distinguishing real vs fake images.
    Takes images of size (1, 768, 768) and outputs logits.
    Output logits: positive = real, negative = fake
    Use with BCEWithLogitsLoss for training.
    """
    
    def __init__(self, input_channels=3, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        self.use_spectral_norm = use_spectral_norm
        
        # Helper function to optionally apply spectral normalization
        def maybe_spectral_norm(layer):
            if self.use_spectral_norm:
                return nn.utils.spectral_norm(layer)
            return layer
        
        # Convolutional layers with increasing depth
        # Input: (batch, 1, 768, 768)
        self.conv1 = maybe_spectral_norm(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)
        )  # -> (batch, 64, 384, 384)
        
        self.conv2 = maybe_spectral_norm(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        )  # -> (batch, 128, 192, 192)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = maybe_spectral_norm(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        )  # -> (batch, 256, 96, 96)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = maybe_spectral_norm(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        )  # -> (batch, 512, 48, 48)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.conv5 = maybe_spectral_norm(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        )  # -> (batch, 512, 24, 24)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.conv6 = maybe_spectral_norm(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        )  # -> (batch, 512, 12, 12)
        self.bn6 = nn.BatchNorm2d(512)
        
        # Adaptive pooling to handle any remaining spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = maybe_spectral_norm(nn.Linear(512 * 4 * 4, 1024))
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = maybe_spectral_norm(nn.Linear(1024, 256))
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer (logits, no sigmoid)
        self.fc_out = maybe_spectral_norm(nn.Linear(256, 1))
        
        # LeakyReLU for all activations
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        """
        Forward pass through discriminator
        
        Args:
            x: Input image tensor of shape (batch_size, 1, 768, 768)
            
        Returns:
            Logits of shape (batch_size, 1) where:
            - Positive values indicate real images
            - Negative values indicate fake images
        """
        # Convolutional layers
        x = self.leaky_relu(self.conv1(x))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leaky_relu(x)
        
        # Adaptive pooling and flatten
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.dropout2(x)
        
        # Output logits (no sigmoid)
        logits = self.fc_out(x)
        
        return logits