import torch

def gaussian_attack(img, std=0.05):
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, -1, 1)