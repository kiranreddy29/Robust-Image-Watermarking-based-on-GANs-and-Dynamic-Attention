import torch
import torch.nn as nn

class Gaussian_Noise(nn.Module):
    def __init__(self, mean, sigma):
        super(Gaussian_Noise, self).__init__()
        self.mean = float(mean)
        self.sigma = float(sigma)

    def forward(self, noise_and_cover):
        encode_image, cover_image = noise_and_cover
        B, C, H, W = encode_image.shape
        
        noise = torch.randn((B, 1, H, W), device=encode_image.device) * self.sigma + self.mean
        noise = torch.clamp(noise, 0.0, 1.0)
        
        noised_image = encode_image + noise
        
        return [noised_image, cover_image]