import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock as TimmSwinBlock

_SWIN_DIM   = 96
_SWIN_HEADS = 3
_INPUT_RES  = (224, 224)


class InvertibleBlock(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        half = in_channels // 2
        self.s1 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half, 3, padding=1), nn.Tanh())
        self.t1 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half, 3, padding=1))
        self.s2 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half, 3, padding=1), nn.Tanh())
        self.t2 = nn.Sequential(nn.Conv2d(half, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, half, 3, padding=1))

    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        if not reverse:
            y2 = x2 * torch.exp(self.s1(x1)) + self.t1(x1)
            y1 = x1
            z1 = y1 * torch.exp(self.s2(y2)) + self.t2(y2)
            z2 = y2
            return torch.cat([z1, z2], dim=1)
        else:
            z1, z2 = x1, x2
            y2 = z2
            y1 = (z1 - self.t2(y2)) * torch.exp(-self.s2(y2))
            x1 = y1
            x2 = (y2 - self.t1(x1)) * torch.exp(-self.s1(x1))
            return torch.cat([x1, x2], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, channels=3, growth_rate=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.conv2(torch.cat([x, out1], dim=1))
        return out2 + x


class DynamicMLP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(channels, channels // 2)
        self.relu    = nn.ReLU(inplace=True)
        self.fc2     = nn.Linear(channels // 2, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x)
        w = self.flatten(w)
        w = self.relu(self.fc1(w))
        w = self.fc2(w)
        return w.view(b, c, 1, 1)


class SwinBlock(nn.Module):
    def __init__(self, channels, window_size=4):
        super().__init__()
        self.proj_in     = nn.Conv2d(channels, _SWIN_DIM, kernel_size=1)
        self.swin        = TimmSwinBlock(
            dim              = _SWIN_DIM,
            input_resolution = _INPUT_RES,
            num_heads        = _SWIN_HEADS,
            window_size      = window_size,
            shift_size       = window_size // 2,
        )
        self.proj_out    = nn.Conv2d(_SWIN_DIM, channels, kernel_size=1)
        self.dynamic_mlp = DynamicMLP(channels)

    def forward(self, x):
        feat = self.proj_in(x)
        feat = feat.permute(0, 2, 3, 1)
        feat = self.swin(feat)
        feat = feat.permute(0, 3, 1, 2)
        feat = self.proj_out(feat)
        return x + feat * self.dynamic_mlp(x)

class EnhancementModule(nn.Module):
    def __init__(self, channels=3, window_size=4):
        super().__init__()
        self.pre_dense  = DenseBlock(channels)
        self.swin_block = SwinBlock(channels, window_size=window_size)
        self.post_dense = DenseBlock(channels)

    def forward(self, x):
        feat = self.pre_dense(x)
        feat = self.swin_block(feat)
        feat = self.post_dense(feat)
        return feat + x


class DifferentialFeatureExtractor(nn.Module):
    def __init__(self, channels=3, growth_rate=32):
        super().__init__()
        ch = channels
        self.conv1 = nn.Conv2d(ch,                 growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(ch +   growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(ch + 2*growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(ch + 3*growth_rate, growth_rate, 3, padding=1)
        self.final = nn.Conv2d(ch + 4*growth_rate, ch,          1)

    def forward(self, xc, xd):
        diff = xc - xd
        x1   = F.relu(self.conv1(diff))
        x2   = F.relu(self.conv2(torch.cat([diff, x1],          dim=1)))
        x3   = F.relu(self.conv3(torch.cat([diff, x1, x2],     dim=1)))
        x4   = F.relu(self.conv4(torch.cat([diff, x1, x2, x3], dim=1)))
        return self.final(torch.cat([diff, x1, x2, x3, x4],    dim=1))


class WatermarkGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.isn          = InvertibleBlock(in_channels=6)
        self.enhance_pre  = EnhancementModule(channels=3, window_size=4)
        self.enhance_post = EnhancementModule(channels=3, window_size=8)
        self.diff_feat    = DifferentialFeatureExtractor(channels=3)

    def embed(self, cover, secret):
        isn_out     = self.isn(torch.cat([cover, secret], dim=1), reverse=False)
        watermarked = isn_out[:, :3, :, :]
        return torch.clamp(watermarked, -1.0, 1.0)

    def extract(self, attacked, watermarked):
        xc_feat     = self.diff_feat(watermarked, attacked)
        xd_enhanced = self.enhance_pre(attacked)
        isn_out     = self.isn(torch.cat([xd_enhanced, xc_feat], dim=1), reverse=True)
        raw_secret  = isn_out[:, 3:, :, :]
        return torch.clamp(self.enhance_post(raw_secret), -1.0, 1.0)

    def forward(self, cover, secret):
        return self.embed(cover, secret)