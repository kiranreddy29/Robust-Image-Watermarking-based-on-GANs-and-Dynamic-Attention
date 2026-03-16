import torch
import torch.nn as nn
from models.generator import WatermarkGenerator
from models.discriminator import Discriminator
from utils.dataset import get_loader
from utils.attacks import gaussian_attack
from utils.metrics import psnr

# -------------------------
# Device
# -------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# Models
# -------------------------
G = WatermarkGenerator().to(device)
D = Discriminator().to(device)

# -------------------------
# Optimizers
# -------------------------
opt_g = torch.optim.Adam(G.parameters(), lr=1e-4)
opt_d = torch.optim.Adam(D.parameters(), lr=1e-4)

# -------------------------
# Loss
# -------------------------
bce = nn.BCELoss()
mse = nn.MSELoss()

# -------------------------
# Data
# -------------------------
cover_loader = get_loader("data/DIV2K/cover", batch_size=1)
wm_loader = get_loader("data/DIV2K/watermark", batch_size=1)

# -------------------------
# Training params
# -------------------------
epochs = 5

# -------------------------
# Training loop
# -------------------------
for epoch in range(epochs):
    for (cover, _), (wm, _) in zip(cover_loader, wm_loader):

        cover = cover.to(device)
        wm = wm.to(device)

        # =====================================================
        # 1. Generator forward
        # =====================================================
        watermarked = G(cover, wm)
        attacked = gaussian_attack(watermarked)

        # =====================================================
        # 2. Train Discriminator
        # =====================================================
        D_real = D(cover)
        D_fake = D(attacked.detach())

        real_labels = torch.ones_like(D_real)
        fake_labels = torch.zeros_like(D_fake)

        d_loss = bce(D_real, real_labels) + bce(D_fake, fake_labels)

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # =====================================================
        # 3. Train Generator
        # =====================================================
        D_fake_for_G = D(attacked)
        g_adv_loss = bce(D_fake_for_G, real_labels)

        # Optional image fidelity loss (helps PSNR)
        g_img_loss = mse(watermarked, cover)

        # Final generator loss
        g_loss = g_adv_loss + g_img_loss

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    # =====================================================
    # 4. Metrics (end of epoch)
    # =====================================================
    with torch.no_grad():
        psnr_c = psnr(watermarked, cover)

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"D Loss: {d_loss.item():.4f} | "
        f"G Loss: {g_loss.item():.4f} | "
        f"PSNR-C: {psnr_c:.2f} dB"
    )

print("Training finished.")
torch.save(G.state_dict(), "generator.pth")
print("Generator model saved.")