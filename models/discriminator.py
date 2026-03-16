import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.mean(dim=[2, 3])  # Global Average Pooling
        return torch.sigmoid(self.classifier(feat))