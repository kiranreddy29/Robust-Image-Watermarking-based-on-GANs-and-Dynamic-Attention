import torch
from models.generator import WatermarkGenerator
from utils.dataset import get_loader
from utils.attacks import gaussian_attack
from utils.metrics import psnr

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
G = WatermarkGenerator().to(device)
G.load_state_dict(
    torch.load("generator.pth", map_location=device, weights_only=True)
)
G.eval()

# Load data
cover_loader = get_loader("data/DIV2K/cover")
wm_loader = get_loader("data/DIV2K/watermark")

cover, _ = next(iter(cover_loader))
wm, _ = next(iter(wm_loader))

cover = cover.to(device)
wm = wm.to(device)

# Forward
with torch.no_grad():
    watermarked = G(cover, wm)
    attacked = gaussian_attack(watermarked)

# Metrics
psnr_c = psnr(watermarked, cover)
print("PSNR-C (Cover vs Watermarked):", psnr_c)
print("Cover range:", cover.min().item(), cover.max().item())
print("Watermarked range:", watermarked.min().item(), watermarked.max().item())