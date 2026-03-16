import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Simplified Invertible Block (placeholder)
# -------------------------
class InvertibleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Dense Block (lightweight)
# -------------------------
class DenseBlock(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return out + x  # dense-style skip

# -------------------------
# Lightweight Swin-style Attention (placeholder)
# -------------------------
class SwinTransformerBlock(nn.Module):
    def __init__(self, window_size=4):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        # NOT full Swin – lightweight spatial attention
        weights = torch.sigmoid(self.attn(x))
        return x * weights

# -------------------------
# Dynamic Attention Module
# -------------------------
class DynamicAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_dense = DenseBlock()
        self.swin_block = SwinTransformerBlock(window_size=4)
        self.post_dense = DenseBlock()

    def forward(self, x):
        feat = self.pre_dense(x)
        feat = self.swin_block(feat)
        feat = self.post_dense(feat)
        return feat + x

# -------------------------
# Generator
# -------------------------
class WatermarkGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.isn_embedding = InvertibleBlock(in_channels=6)
        self.enhance_module = DynamicAttentionModule()

    def forward(self, cover_image, secret_image):
        x = torch.cat([cover_image, secret_image], dim=1)
        watermarked = self.isn_embedding(x)
        enhanced = self.enhance_module(watermarked)
        return torch.clamp(enhanced, -1, 1)