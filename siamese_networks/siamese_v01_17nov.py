import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the input channel size parameter to the SiameseNetwork class

class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity comparison.
    Takes two images of size (1, 768, 768) and outputs probability.
    Output: 0 = similar images, 1 = dissimilar images
    """
    
    def __init__(self, input_channels=3):
        super(SiameseNetwork, self).__init__()
        
        # Shared CNN backbone for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=10, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to handle any input size variations
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Embedding layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        
        # Final classification layer
        self.fc_out = nn.Linear(256, 1)
        
    def forward_once(self, x):
        """Forward pass through one branch of the network to get embedding"""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        
        return x
    
    def forward(self, img1, img2):
        """
        Forward pass through both branches and compute similarity probability
        
        Args:
            img1: First image tensor of shape (batch_size, 1, 768, 768)
            img2: Second image tensor of shape (batch_size, 1, 768, 768)
            
        Returns:
            Probability tensor of shape (batch_size, 1) where:
            - 0 indicates similar images
            - 1 indicates dissimilar images
        """
        # Get embeddings from both images
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        
        # Compute absolute difference between embeddings
        difference = torch.abs(embedding1 - embedding2)
        
        # Pass through final layer and sigmoid to get probability
        output = self.fc_out(difference)
        probability = torch.sigmoid(output)
        
        return probability