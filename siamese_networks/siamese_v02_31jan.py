import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    Improved Siamese Network for image similarity comparison.
    Takes two images of size (3, 512, 512) and outputs probability.
    Output: 0 = similar images, 1 = dissimilar images
    """
    
    def __init__(self, input_channels=3, embedding_dim=256):
        super(SiameseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Shared CNN backbone for feature extraction (optimized for 512x512)
        # Input: 3 x 512 x 512
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Output: 64 x 128 x 128
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 128 x 32 x 32
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 256 x 16 x 16
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 512 x 8 x 8
        
        # Global Average Pooling instead of adaptive pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Improved embedding layers with residual connection concept
        self.fc1 = nn.Linear(512, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, embedding_dim)
        self.bn_fc2 = nn.BatchNorm1d(embedding_dim)
        
        # Distance metric layer (processes the combined features)
        self.fc_distance1 = nn.Linear(embedding_dim, 128)
        self.bn_distance = nn.BatchNorm1d(128)
        self.dropout_distance = nn.Dropout(0.3)
        
        self.fc_distance2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward_once(self, x):
        """Forward pass through one branch of the network to get embedding"""
        # Convolutional layers with batch norm and ReLU
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Double conv layers for better feature extraction
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(F.relu(self.bn3_2(self.conv3_2(x))))
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(F.relu(self.bn4_2(self.conv4_2(x))))
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 512)
        
        # Fully connected layers with batch norm
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Final embedding (L2 normalized for better distance metrics)
        x = self.bn_fc2(self.fc2(x))
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        
        return x
    
    def forward(self, img1, img2):
        """
        Forward pass through both branches and compute similarity probability
        
        Args:
            img1: First image tensor of shape (batch_size, 3, 512, 512)
            img2: Second image tensor of shape (batch_size, 3, 512, 512)
            
        Returns:
            Probability tensor of shape (batch_size, 1) where:
            - Values close to 0 indicate similar images
            - Values close to 1 indicate dissimilar images
        """
        # Get L2-normalized embeddings from both images
        embedding1 = self.forward_once(img1)
        embedding2 = self.forward_once(img2)
        
        # Compute absolute difference between embeddings
        difference = torch.abs(embedding1 - embedding2)
        
        # Process the difference through distance metric layers
        x = F.relu(self.bn_distance(self.fc_distance1(difference)))
        x = self.dropout_distance(x)
        x = F.relu(self.fc_distance2(x))
        
        # Final output with sigmoid activation
        output = self.fc_out(x)
        probability = torch.sigmoid(output)
        
        return probability
    
