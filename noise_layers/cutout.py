import torch
import torch.nn as nn
import random

class CutoutAttack(nn.Module):
    def __init__(self, drop_prob=0.15, block_size=48):
        super(CutoutAttack, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, noised_and_cover):
        image = noised_and_cover[0].clone()
        b, c, h, w = image.shape
        mask = torch.ones_like(image)
        num_blocks = int((h * w * self.drop_prob) / (self.block_size ** 2))

        for batch_idx in range(b):
            for _ in range(num_blocks):
                x = random.randint(0, w - self.block_size)
                y = random.randint(0, h - self.block_size)
                mask[batch_idx, :, y:y+self.block_size, x:x+self.block_size] = 0.0

        noised_image = image * mask
        return [noised_image, noised_and_cover[1]]