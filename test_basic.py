import torch
from models.generator import WatermarkGenerator
from utils.dataset import get_loader
from noise_layers.Gaussian_noise import Gaussian_Noise
from utils.metrics import psnr

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {device}")

G = WatermarkGenerator().to(device)

try:
    G.load_state_dict(torch.load("generator_final.pth", map_location=device, weights_only=True))
    print("Loaded trained weights successfully!")
except FileNotFoundError:
    print("No trained weights found. Running with initialized parameters.")

G.eval()

cover_loader = get_loader("data/DIV2K/cover", batch_size=1, shuffle=False)
wm_loader = get_loader("data/DIV2K/watermark", batch_size=1, shuffle=False)

cover_batch = next(iter(cover_loader))
wm_batch = next(iter(wm_loader))

cover = cover_batch[0].to(device) if isinstance(cover_batch, (list, tuple)) else cover_batch.to(device)
wm = wm_batch[0].to(device) if isinstance(wm_batch, (list, tuple)) else wm_batch.to(device)

print("\n--- Running Pipeline Test ---")
with torch.no_grad():
    watermarked = G(cover, wm)
    
    attack_layer = Gaussian_Noise(mean=0.0, sigma=10.0/127.5).to(device)
    attacked = attack_layer([watermarked.clone(), cover.clone()])[0]
    
    extracted_wm = G.extract(attacked, watermarked)

psnr_c = psnr(watermarked, cover)
psnr_s = psnr(extracted_wm, wm)

print("\n--- Results ---")
print(f"PSNR-C (Cover vs Watermarked): {psnr_c:.2f} dB")
print(f"PSNR-S (Secret vs Extracted):  {psnr_s:.2f} dB")
print("Pipeline test completed successfully!")