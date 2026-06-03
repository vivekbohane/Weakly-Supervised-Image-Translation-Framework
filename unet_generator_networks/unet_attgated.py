import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Attention Gate as described in the paper"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        # Gating signal convolution (from decoder)
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Input feature convolution (from encoder/skip connection)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention coefficient convolution
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        g: gating signal from decoder (coarser scale)
        x: input features from encoder (skip connection)
        """
        # Transform both inputs to intermediate dimension
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample gating signal if needed to match x spatial dimensions
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        # Additive attention
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention coefficients
        return x * psi, psi


class DoubleConv(nn.Module):
    """Double Convolution block with dropout"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with channel attention"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with both Attention Gate and Channel Attention"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        
        # Attention Gate: F_g (decoder features), F_l (skip features), F_int (intermediate)
        self.attention_gate = AttentionGate(
            F_g=in_channels // 2,
            F_l=in_channels // 2,
            F_int=out_channels
        )
        
        self.conv = DoubleConv(in_channels, out_channels, dropout_rate)
    
    def forward(self, x, skip):
        # Upsample decoder features
        x = self.up(x)
        
        # Apply Attention Gate to skip connection
        skip_attended, attention_weights = self.attention_gate(g=x, x=skip)
        
        # Concatenate upsampled features with gated skip connection
        x = torch.cat([skip_attended, x], dim=1)
        
        # Apply convolutions
        x = self.conv(x)
        
        return x , attention_weights


class UNet(nn.Module):
    """U-Net with Skip Connections, Channel Attention, Attention Gates, and Dropout"""
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], dropout_rate=0.1):
        super(UNet, self).__init__()
        
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, features[0], dropout_rate)
        self.encoder2 = EncoderBlock(features[0], features[1], dropout_rate)
        self.encoder3 = EncoderBlock(features[1], features[2], dropout_rate)
        self.encoder4 = EncoderBlock(features[2], features[3], dropout_rate)
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[3], features[3] * 2, dropout_rate)
        
        # Decoder path (now with Attention Gates)
        self.decoder4 = DecoderBlock(features[3] * 2, features[3], dropout_rate)
        self.decoder3 = DecoderBlock(features[3], features[2], dropout_rate)
        self.decoder2 = DecoderBlock(features[2], features[1], dropout_rate)
        self.decoder1 = DecoderBlock(features[1], features[0], dropout_rate)
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder path with skip connections
        x, skip1 = self.encoder1(x)  # 256 -> 128
        x, skip2 = self.encoder2(x)  # 128 -> 64
        x, skip3 = self.encoder3(x)  # 64 -> 32
        x, skip4 = self.encoder4(x)  # 32 -> 16
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with attention-gated skip connections
        x , att_weights_4 = self.decoder4(x, skip4)  # 16 -> 32
        x , att_weights_3 = self.decoder3(x, skip3)  # 32 -> 64
        x , att_weights_2 = self.decoder2(x, skip2)  # 64 -> 128
        x , att_weights_1 = self.decoder1(x, skip1)  # 128 -> 256
        
        # Final output
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x , att_weights_1  # Return last attention weights for visualization                 

# # Example usage
# if __name__ == "__main__":
#     # Create combined model
#     model = UNet()
    
#     # Test with random input
#     x = torch.randn(1, 1, 256, 256)
#     output , att_map = model(x)
    
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Attention map shape: {att_map.shape}")
#     print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")   