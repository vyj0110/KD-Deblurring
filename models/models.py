import torch
import torch.nn as nn
import torch.nn.functional as F

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) block for adaptive channel-wise feature recalibration.

    Parameters:
    - channels (int): Number of channels in the input feature map.
    - gamma (int): Hyperparameter to control kernel size calculation.
    - b (int): Bias term for kernel size calculation.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Determine kernel size based on channel dimension
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1  # Ensure kernel size is odd

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply global average pooling across spatial dimensions
        y = self.avg_pool(x)

        # Apply 1D convolution across channels
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Generate channel-wise weights and apply to input
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """
    Convolutional block with two 3x3 Conv-BN-ReLU layers and optional ECA.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - use_eca (bool): Whether to use Efficient Channel Attention (ECA).
    """
    def __init__(self, in_channels, out_channels, use_eca=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.eca = ECABlock(out_channels) if use_eca else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.eca(x)

class LightweightUNet(nn.Module):
    """
    Lightweight U-Net with ECA modules and multiscale input fusion for image restoration tasks.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.downsample = nn.AvgPool2d(kernel_size=2)

        # Encoder path
        self.enc1 = ConvBlock(in_channels * 2, 64)  # input + downsampled input
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder path
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output projection
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Multiscale input fusion: concatenate input with its downsampled version
        x_down = self.downsample(x)
        x_fused = torch.cat([
            x,
            F.interpolate(x_down, size=x.shape[2:], mode='bilinear', align_corners=False)
        ], dim=1)

        # Encoder
        s1 = self.enc1(x_fused)
        x = self.pool1(s1)
        s2 = self.enc2(x)
        x = self.pool2(s2)
        s3 = self.enc3(x)
        x = self.pool3(s3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.upconv3(x)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)

        x = self.upconv2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = self.upconv1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)

        # Output projection with clamping to [0, 1] range
        return torch.clamp(self.out_conv(x), 0.0, 1.0)
