import torch
from models.generator import WatermarkGenerator, DifferentialFeatureExtractor
from utils.dataset import get_loader
from noise_layers.Gaussian_noise import Gaussian_Noise
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

# Unpack the list into image and dummy label
cover, _ = next(iter(cover_loader))
wm, _ = next(iter(wm_loader))

# Now push just the image tensors to the device
cover = cover.to(device)
wm = wm.to(device)

# -------------------------
# 3. Test Pipeline
# -------------------------
print("\n--- Running Pipeline Test ---")
with torch.no_grad():
    # Step A: Embed Watermark
    print("1. Embedding watermark...")
    watermarked = G(cover, wm)
    
    # Step B: Apply Attack
    print("2. Applying distortion attack...")
    # Initialize the attack with mean=0 and sigma=10 (as tested in the paper)
    attack_layer = Gaussian_Noise(mean=0.0, sigma=10.0).to(device)
    
    # Pass the image as a list, and grab the 0th index from the output
    attacked_input = [watermarked.clone(), cover.clone()]
    attacked = attack_layer(attacked_input)[0]
    
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