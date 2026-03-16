import torch
from models.generator import WatermarkGenerator, DifferentialFeatureExtractor
from utils.dataset import get_loader
from utils.attacks import gaussian_attack
from utils.metrics import psnr

# -------------------------
# Device Configuration
# -------------------------
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {device}")

# -------------------------
# 1. Initialize Models
# -------------------------
G = WatermarkGenerator().to(device)
DiffExtractor = DifferentialFeatureExtractor(channels=3).to(device)

# Load pre-trained weights if they exist (will skip if you haven't trained yet)
try:
    G.load_state_dict(torch.load("generator_2026.pth", map_location=device, weights_only=True))
    DiffExtractor.load_state_dict(torch.load("diff_extractor_2026.pth", map_location=device, weights_only=True))
    print("Loaded trained weights successfully.")
except FileNotFoundError:
    print("No trained weights found. Running with initialized parameters for pipeline test.")

G.eval()
DiffExtractor.eval()

# -------------------------
# 2. Load Data
# -------------------------
cover_loader = get_loader("data/DIV2K/cover", batch_size=1)
wm_loader = get_loader("data/DIV2K/watermark", batch_size=1)

# The dataloader yields a list: [image_tensor, label_tensor]
cover_batch = next(iter(cover_loader))
wm_batch = next(iter(wm_loader))

# Grab the 0th index (the actual images) and send them to the device
cover = cover_batch[0].to(device)
wm = wm_batch[0].to(device)

# -------------------------
# 3. Test Pipeline
# -------------------------
print("\n--- Running Pipeline Test ---")
with torch.no_grad():
    # Step A: Embed Watermark
    print("1. Embedding watermark...")
    watermarked = G(cover, wm)
    
    # Step B: Apply Attack [cite: 651, 654]
    print("2. Applying distortion attack...")
    attacked = gaussian_attack(watermarked, std=0.1)
    
    # Step C: Extract Differential Features
    print("3. Extracting differential features...")
    diff_features = DiffExtractor(watermarked, attacked)
    
    # Step D: Recover Watermark
    print("4. Recovering watermark...")
    extracted_wm = G.extract(attacked, diff_features)

# -------------------------
# 4. Compute Metrics
# -------------------------
psnr_c = psnr(watermarked, cover)
psnr_s = psnr(extracted_wm, wm)

print("\n--- Results ---")
print(f"PSNR-C (Cover vs Watermarked): {psnr_c:.2f} dB")
print(f"PSNR-S (Secret vs Extracted):  {psnr_s:.2f} dB")
print(f"Watermarked tensor shape: {watermarked.shape}")
print(f"Extracted tensor shape:   {extracted_wm.shape}")
print("Pipeline test completed successfully!")