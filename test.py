import torch
from models.generator import WatermarkGenerator
from utils.dataset import get_loader
from utils.metrics import psnr

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

G = WatermarkGenerator().to(device)
G.load_state_dict(torch.load("generator_final.pth", map_location=device, weights_only=True))
G.eval()

cover_loader = get_loader("data/DIV2K/cover", batch_size=1, shuffle=False)
wm_loader    = get_loader("data/DIV2K/watermark", batch_size=1, shuffle=False)

psnr_c_total = 0
count = 0
with torch.no_grad():
    for cover_batch, wm_batch in zip(cover_loader, wm_loader):
        cover = cover_batch[0].to(device) if isinstance(cover_batch, (list, tuple)) else cover_batch.to(device)
        wm = wm_batch[0].to(device) if isinstance(wm_batch, (list, tuple)) else wm_batch.to(device)
        
        watermarked = G(cover, wm)
        psnr_c_total += psnr(watermarked, cover)
        count += 1

print(f"Avg PSNR-C over {count} images: {psnr_c_total/count:.2f} dB")