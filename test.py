import torch
import os
from models.generator import WatermarkGenerator
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.identity import Identity
from utils.dataset import get_loader
from utils.metrics import psnr

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def evaluate():
    G = WatermarkGenerator().to(device)
    
    if os.path.exists("generator_final.pth"):
        G.load_state_dict(torch.load("generator_final.pth", map_location=device, weights_only=True))
    else:
        print("Warning: generator_final.pth not found. Running with initialized parameters.")
        
    G.eval()

    cover_loader = get_loader("data/DIV2K/cover", batch_size=1, shuffle=False)
    wm_loader    = get_loader("data/DIV2K/watermark", batch_size=1, shuffle=False)

    attacks = {
        "No Attack": Identity().to(device),
        "Gaussian (σ=10)": Gaussian_Noise(mean=0.0, sigma=10.0/127.5).to(device),
        "JPEG Compression": JpegCompression(device).to(device)
    }

    results = {name: {"psnr_c": 0.0, "psnr_s": 0.0} for name in attacks.keys()}
    count = 0

    print("Running evaluation on dataset...")
    
    with torch.no_grad():
        for cover, wm in zip(cover_loader, wm_loader):
            cover = cover.to(device)
            wm = wm.to(device)
            
            watermarked = G(cover, wm)
            
            for attack_name, attack_layer in attacks.items():
                attacked = attack_layer([watermarked.clone(), cover.clone()])[0]
                extracted_wm = G.extract(attacked, watermarked)
                
                results[attack_name]["psnr_c"] += psnr(watermarked, cover)
                results[attack_name]["psnr_s"] += psnr(extracted_wm, wm)
            
            count += 1

    print(f"\nEvaluation Results over {count} images:")
    for attack_name in attacks.keys():
        avg_c = results[attack_name]["psnr_c"] / count
        avg_s = results[attack_name]["psnr_s"] / count
        print(f"[{attack_name}] PSNR-C: {avg_c:.2f} dB | PSNR-S: {avg_s:.2f} dB")

if __name__ == "__main__":
    evaluate()