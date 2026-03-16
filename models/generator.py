import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Simplified Invertible Block (placeholder)
# -------------------------
class InvertibleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # INNs split the input in half, so channels must be even
        assert in_channels % 2 == 0
        half_channels = in_channels // 2
        
        # Sub-networks for scale (s) and translation (t)
        self.subnet_s = nn.Sequential(
            nn.Conv2d(half_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, half_channels, 3, padding=1),
            nn.Tanh() # Tanh keeps scaling stable
        )
        self.subnet_t = nn.Sequential(
            nn.Conv2d(half_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, half_channels, 3, padding=1)
        )

    def forward(self, x, reverse=False):
        # Split input into two halves along the channel dimension
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        if not reverse:
            # Forward pass (Embedding the watermark)
            s = self.subnet_s(x1)
            t = self.subnet_t(x1)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            return torch.cat([y1, y2], dim=1)
        else:
            # Reverse pass (Extracting the watermark)
            s = self.subnet_s(x1)
            t = self.subnet_t(x1)
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([y1, y2], dim=1)

# -------------------------
# Dense Block (lightweight)
# -------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels=3, growth_rate=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, in_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        return out2 + x 

class DynamicMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.net(x).view(b, c, 1, 1)
        return w

class SwinTransformerBlock(nn.Module):
    def __init__(self, channels, window_size=4):
        super().__init__()
        self.window_size = window_size
        self.conv_global = nn.Conv2d(channels, channels, 3, padding=1)
        self.dynamic_mlp = DynamicMLP(channels)

    def forward(self, x):
        global_feat = F.relu(self.conv_global(x))
        dynamic_weights = self.dynamic_mlp(x)
        weighted_feat = global_feat * dynamic_weights
        return x + weighted_feat

class DynamicAttentionModule(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.pre_dense = DenseBlock(channels)
        self.swin_block_4x4 = SwinTransformerBlock(channels, window_size=4)
        self.swin_block_8x8 = SwinTransformerBlock(channels, window_size=8)
        self.post_dense = DenseBlock(channels)

    def forward(self, x):
        feat = self.pre_dense(x)
        feat = self.swin_block_4x4(feat)
        feat = self.swin_block_8x8(feat)
        feat = self.post_dense(feat)
        return feat + x
    
class DifferentialFeatureExtractor(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + 32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 96, 32, 3, padding=1)
        self.final_conv = nn.Conv2d(channels + 128, channels, 1)

    def forward(self, xc, xd):
        diff = xc - xd
        x1 = F.relu(self.conv1(diff))
        x2 = F.relu(self.conv2(torch.cat([diff, x1], dim=1)))
        x3 = F.relu(self.conv3(torch.cat([diff, x1, x2], dim=1)))
        x4 = F.relu(self.conv4(torch.cat([diff, x1, x2, x3], dim=1)))
        out = self.final_conv(torch.cat([diff, x1, x2, x3, x4], dim=1))
        return out
# -------------------------
# Generator
# -------------------------
class WatermarkGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.isn_embedding = InvertibleBlock(in_channels=6)
        self.enhance_module = DynamicAttentionModule()

    def forward(self, cover_image, secret_image):
        # Concatenate inputs to 6 channels
        x = torch.cat([cover_image, secret_image], dim=1)
        
        # ISN outputs 6 channels
        isn_out = self.isn_embedding(x)
        
        # SLICE: Extract only the first 3 channels (the watermarked image)
        watermarked = isn_out[:, :3, :, :]
        
        # Enhance the 3-channel watermarked image
        enhanced = self.enhance_module(watermarked)
        return torch.clamp(enhanced, -1, 1)

    def extract(self, attacked_image, diff_features):
        # Concatenate attacked image and diff features to 6 channels
        combined_input = torch.cat([attacked_image, diff_features], dim=1)
        
        # Reverse ISN outputs 6 channels
        extracted_raw = self.isn_embedding(combined_input, reverse=True)
        
        # SLICE: The recovered secret is in the second half of the channels
        recovered_secret = extracted_raw[:, 3:, :, :]
        
        # Apply post-extraction enhancement
        recovered_secret = self.enhance_module.post_dense(recovered_secret)
        
        return torch.clamp(recovered_secret, -1, 1)
    
class WatermarkExtractor(nn.Module):
    """Extracts secret image from (attacked) watermarked image"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3,  3, padding=1), nn.Tanh()
        )
    def forward(self, watermarked_or_attacked):
        return self.net(watermarked_or_attacked)