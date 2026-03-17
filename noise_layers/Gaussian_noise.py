import numpy as np
import torch.nn as nn
import torch

class Gaussian_Noise(nn.Module):
    def __init__(self, mean, sigma):
        super(Gaussian_Noise, self).__init__()
        self.mean = float(mean)
        self.sigma = float(sigma)
        # Removed the hardcoded device check from here

    def forward(self, noise_and_cover):
        encode_image = noise_and_cover[0]
        
        # Dynamically grab the exact device the image is currently on (e.g., 'mps:0')
        current_device = encode_image.device
        
        B, C, H, W = encode_image.size()
        noise = np.clip(np.random.normal(self.mean, self.sigma, (B, 1, H, W)), 0, 1)
        
        # Create the noise tensor directly on that same device
        noise = torch.tensor(noise, dtype=torch.float32, device=current_device)
        
        for i in range(C):
            encode_image[:, i:, :, :] = encode_image[:, i:, :, :] + noise
            
        noise_and_cover[0] = encode_image
        return noise_and_cover