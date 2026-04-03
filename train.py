import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import psnr
from tqdm import tqdm

from noise_layers.cutout import CutoutAttack
from models.generator import WatermarkGenerator
from models.discriminator import Discriminator
from noise_layers.noiser import Noiser
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.identity import Identity
from utils.dataset import get_loader

# -------------------------------------------------------------------
# THE NUCLEAR OPTION: Forces PyTorch to trace the exact line of any memory leaks
torch.autograd.set_detect_anomaly(True)
# -------------------------------------------------------------------

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

def wavelet_ll_loss(xc, xh):
    ll_xc = F.avg_pool2d(xc, kernel_size=2, stride=2)
    ll_xh = F.avg_pool2d(xh, kernel_size=2, stride=2)
    return F.mse_loss(ll_xc, ll_xh)

def discriminator_loss(d_real, d_fake):
    return -(
        torch.mean(torch.log(d_real + 1e-8)) +
        torch.mean(torch.log(1.0 - d_fake + 1e-8))
    )

def generator_adv_loss(d_fake):
    return -torch.mean(torch.log(d_fake + 1e-8))

EPOCHS         = 1600
BATCH_SIZE     = 4
PHASE2_EPOCH   = 300
PHASE3_EPOCH   = 1000
LR_EARLY       = 10 ** -4.5
LR_LATE        = 10 ** -5.5
LAMBDA_ADV     = 0.1
LAMBDA_F       = 0.05
LAMBDA_C       = 1.0
LAMBDA_S       = 1.0
GAUSSIAN_SIGMA = 10.0 / 127.5
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    G = WatermarkGenerator().to(device)
    
    D = Discriminator().to(device)         
    D_secret = Discriminator().to(device)  

    noise_list    = [
        Gaussian_Noise(mean=0.0, sigma=GAUSSIAN_SIGMA), 
        'JpegPlaceholder', 
        'QuantizationPlaceholder', 
        CutoutAttack(drop_prob=0.15, block_size=48), 
        Identity()
    ]
    attack_module = Noiser(noise_list, device=device).to(device)

    mse_loss = nn.MSELoss()

    opt_g = torch.optim.Adam(G.parameters(), lr=LR_EARLY)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_EARLY)
    opt_d_secret = torch.optim.Adam(D_secret.parameters(), lr=LR_EARLY)

    cover_loader = get_loader("data/DIV2K/cover",     batch_size=BATCH_SIZE, shuffle=True)
    wm_loader    = get_loader("data/DIV2K/watermark", batch_size=BATCH_SIZE, shuffle=True)

    print("Starting Dual-Discriminator training ...")

    for epoch in range(EPOCHS):

        if epoch == PHASE3_EPOCH:
            for pg in opt_g.param_groups: pg['lr'] = LR_LATE
            for pg in opt_d.param_groups: pg['lr'] = LR_LATE
            for pg in opt_d_secret.param_groups: pg['lr'] = LR_LATE
            print(f"\n[Epoch {epoch+1}] Phase 3 LR = {LR_LATE:.2e}")

        G.train()
        D.train()
        D_secret.train()

        epoch_d_loss = 0.0
        epoch_d_sec_loss = 0.0
        epoch_g_loss = 0.0
        epoch_psnr_c = 0.0
        epoch_psnr_s = 0.0
        n_batches    = 0

        pbar = tqdm(
            zip(cover_loader, wm_loader),
            total=min(len(cover_loader), len(wm_loader)),
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            leave=False,
        )

        for cover, wm in pbar:
            if isinstance(cover, (list, tuple)): cover = cover[0]
            if isinstance(wm, (list, tuple)): wm = wm[0]

            cover = cover.to(device)
            wm    = wm.to(device)

            # --- GRAPH ISOLATION BARRIER 1 ---
            watermarked  = G(cover, wm)
            
            # Clone forces the attack module to use fresh memory
            attack_inputs = [watermarked.clone(), cover.clone()]
            attacked = attack_module(attack_inputs)[0]
            attacked = torch.clamp(attacked.clone(), -1.0, 1.0)
            
            # --- GRAPH ISOLATION BARRIER 2 ---
            # Extract uses cloned attacked images so it doesn't bleed backward
            extracted_wm = G.extract(attacked.clone(), watermarked.clone())

            if epoch >= PHASE2_EPOCH:
                # Train Discriminator 1
                opt_d.zero_grad()
                d_real = D(cover.clone())
                d_fake = D(attacked.detach().clone())
                d_loss = discriminator_loss(d_real, d_fake)
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_d.step()
                d_loss_val = d_loss.item()
                
                # Train Discriminator 2
                opt_d_secret.zero_grad()
                d_sec_real = D_secret(wm.clone())
                d_sec_fake = D_secret(extracted_wm.detach().clone())
                d_sec_loss = discriminator_loss(d_sec_real, d_sec_fake)
                d_sec_loss.backward()
                torch.nn.utils.clip_grad_norm_(D_secret.parameters(), max_norm=1.0)
                opt_d_secret.step()
                d_sec_loss_val = d_sec_loss.item()
            else:
                d_loss_val = 0.0
                d_sec_loss_val = 0.0

            opt_g.zero_grad()

            L_pre     = mse_loss(watermarked, cover)
            L_post    = mse_loss(extracted_wm, wm)
            L_enhance = LAMBDA_C * L_pre + LAMBDA_S * L_post
            L_f       = mse_loss(watermarked, attacked)
            L_l       = wavelet_ll_loss(watermarked, cover)

            if epoch < PHASE2_EPOCH:
                g_loss = L_enhance + LAMBDA_F * L_f + L_l
            else:
                # --- GRAPH ISOLATION BARRIER 3 ---
                # Clone inputs so Discriminator parameters don't corrupt Generator graphs
                d_fake_for_g = D(attacked.clone())
                d_sec_fake_for_g = D_secret(extracted_wm.clone())
                
                L_adv  = generator_adv_loss(d_fake_for_g)
                L_adv_sec = generator_adv_loss(d_sec_fake_for_g)
                
                g_loss = L_enhance + (LAMBDA_ADV * L_adv) + (LAMBDA_ADV * L_adv_sec) + LAMBDA_F * L_f + L_l

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_g.step()

            batch_psnr_c  = psnr(watermarked, cover)
            batch_psnr_s  = psnr(extracted_wm, wm)
            epoch_d_loss += d_loss_val
            epoch_d_sec_loss += d_sec_loss_val
            epoch_g_loss += g_loss.item()
            epoch_psnr_c += batch_psnr_c
            epoch_psnr_s += batch_psnr_s
            n_batches    += 1

            pbar.set_postfix({
                "D1": f"{d_loss_val:.3f}",
                "D2": f"{d_sec_loss_val:.3f}",
                "G":  f"{g_loss.item():.3f}",
                "PSNR-C": f"{batch_psnr_c:.1f}",
                "PSNR-S": f"{batch_psnr_s:.1f}",
            })

        avg_d  = epoch_d_loss / n_batches
        avg_g  = epoch_g_loss / n_batches
        avg_pc = epoch_psnr_c / n_batches
        avg_ps = epoch_psnr_s / n_batches

        print(
            f"Epoch [{epoch+1:>4}/{EPOCHS}] | "
            f"D1: {avg_d:.4f} | G: {avg_g:.4f} | "
            f"PSNR-C: {avg_pc:.2f} dB | PSNR-S: {avg_ps:.2f} dB"
        )

        if (epoch + 1) % 100 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1}.pth")
            torch.save({
                "epoch":      epoch + 1,
                "G":          G.state_dict(),
                "D":          D.state_dict(),
                "D_secret":   D_secret.state_dict(),
                "opt_g":      opt_g.state_dict(),
                "opt_d":      opt_d.state_dict(),
                "opt_d_sec":  opt_d_secret.state_dict(),
                "avg_psnr_c": avg_pc,
                "avg_psnr_s": avg_ps,
            }, ckpt_path)
            print(f"  Checkpoint saved -> {ckpt_path}")

    torch.save(G.state_dict(), "generator_final.pth")
    torch.save(D.state_dict(), "discriminator_final.pth")
    torch.save(D_secret.state_dict(), "discriminator_secret_final.pth")

    print("\nTraining complete. Models saved.")

if __name__ == "__main__":
    main()