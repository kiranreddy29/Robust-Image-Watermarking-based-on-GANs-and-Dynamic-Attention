import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ── Model imports ────────────────────────────────────────────────────────────
from models.generator import WatermarkGenerator, DifferentialFeatureExtractor
from models.discriminator import Discriminator

# ── Noise layer imports ───────────────────────────────────────────────────────
from noise_layers.noiser import Noiser
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.identity import Identity

# ─────────────────────────────────────────────────────────────────────────────
# 0. DEVICE  (CUDA > MPS > CPU — fixed order)
# ─────────────────────────────────────────────────────────────────────────────
device = (
    "cuda"  if torch.cuda.is_available()                else
    "mps"   if torch.backends.mps.is_available()        else
    "cpu"
)
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FLAT DATASET  (fixes ImageFolder crash on DIV2K which has no subfolders)
# ─────────────────────────────────────────────────────────────────────────────
class FlatImageDataset(Dataset):
    """Loads all images directly from a flat folder (no class subdirectories)."""
    EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    def __init__(self, folder: str, transform=None):
        self.paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(self.EXTS)
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img   # returns tensor only — NO label tuple


def get_loader(path: str, batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),   # maps [0,1] → [-1,1]
    ])
    dataset = FlatImageDataset(path, transform=tf)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,   # keeps batch sizes consistent
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. WATERMARK EXTRACTOR  (was completely missing — required for PSNR-S)
# ─────────────────────────────────────────────────────────────────────────────
class WatermarkExtractor(nn.Module):
    """
    Recovers the secret image from the (optionally attacked) watermarked image
    plus the differential compensation features.
    Input:  attacked image (3ch) concatenated with diff_features (3ch) → 6ch
    Output: recovered secret image (3ch)
    """
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels,  64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,           64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64,           64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1), nn.Tanh(),
        )

    def forward(self, attacked: torch.Tensor, diff_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([attacked, diff_feat], dim=1)   # (B, 6, H, W)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PSNR  (fixed: .item() so math.sqrt doesn't receive a Tensor)
# ─────────────────────────────────────────────────────────────────────────────
def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR for images normalised to [-1, 1]  (dynamic range = 2.0)."""
    mse = torch.mean((img1.detach() - img2.detach()) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(2.0 / math.sqrt(mse))


# ─────────────────────────────────────────────────────────────────────────────
# 4. WAVELET LOW-FREQUENCY LOSS  (L_l from paper Eq. 8)
# ─────────────────────────────────────────────────────────────────────────────
def wavelet_ll_loss(xc: torch.Tensor, xh: torch.Tensor) -> torch.Tensor:
    """
    Approximate low-frequency (LL) sub-band loss using average pooling.
    A proper Haar DWT LL sub-band = average of 2×2 blocks.
    """
    ll_xc = F.avg_pool2d(xc, kernel_size=2, stride=2)
    ll_xh = F.avg_pool2d(xh, kernel_size=2, stride=2)
    return F.mse_loss(ll_xc, ll_xh)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS        = 1600
BATCH_SIZE    = 16           # reduce to 4 if GPU OOM
PHASE3_EPOCH  = 1000         # switch to lower LR at this epoch
LR_EARLY      = 10 ** -4.5  # ≈ 3.16e-5  (phases 1 & 2)
LR_LATE       = 10 ** -5.5  # ≈ 3.16e-6  (phase 3)
LAMBDA_ADV    = 0.1          # paper Table / Eq. 9
LAMBDA_F      = 0.05         # paper Eq. 9
LAMBDA_C      = 1.0          # paper Section 3.5
LAMBDA_S      = 1.0          # paper Section 3.5
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Paper sigma=10 is in [0,255] pixel range.
# Our images are normalised to [-1,1], so scale: 10/127.5 ≈ 0.0784
GAUSSIAN_SIGMA_NORM = 10.0 / 127.5


# ─────────────────────────────────────────────────────────────────────────────
# 6. MODELS
# ─────────────────────────────────────────────────────────────────────────────
G         = WatermarkGenerator().to(device)
D         = Discriminator().to(device)
Extractor = WatermarkExtractor().to(device)
DiffFeat  = DifferentialFeatureExtractor(channels=3).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 7. ATTACK MODULE
# 'JpegPlaceholder' is valid — Noiser converts it to JpegCompression internally
# Gaussian sigma is scaled to normalised image range
# ─────────────────────────────────────────────────────────────────────────────
noise_list = [
    Gaussian_Noise(mean=0.0, sigma=GAUSSIAN_SIGMA_NORM),
    'JpegPlaceholder',          # Noiser resolves this to JpegCompression(device)
    Identity(),                 # Round-error proxy (no-op, avoids all-attack bias)
]
attack_module = Noiser(noise_list, device=device).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 8. LOSS
# ─────────────────────────────────────────────────────────────────────────────
mse_loss = nn.MSELoss()


def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Standard GAN discriminator loss (Eq. 4 in paper)."""
    return -(
        torch.mean(torch.log(d_real + 1e-8)) +
        torch.mean(torch.log(1.0 - d_fake + 1e-8))
    )


def generator_adv_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Standard GAN generator adversarial loss (Eq. 5 in paper)."""
    return -torch.mean(torch.log(d_fake + 1e-8))


# ─────────────────────────────────────────────────────────────────────────────
# 9. OPTIMIZERS
# ─────────────────────────────────────────────────────────────────────────────
gen_params = (
    list(G.parameters()) +
    list(Extractor.parameters()) +
    list(DiffFeat.parameters())
)
opt_g = torch.optim.Adam(gen_params,       lr=LR_EARLY)
opt_d = torch.optim.Adam(D.parameters(),   lr=LR_EARLY)


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


# ─────────────────────────────────────────────────────────────────────────────
# 10. DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
cover_loader = get_loader("data/DIV2K/cover",     batch_size=BATCH_SIZE, shuffle=True)
wm_loader    = get_loader("data/DIV2K/watermark", batch_size=BATCH_SIZE, shuffle=True)


# ─────────────────────────────────────────────────────────────────────────────
# 11. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("Starting training …")

for epoch in range(EPOCHS):

    # ── Phase-3 learning rate switch ────────────────────────────────────────
    if epoch == PHASE3_EPOCH:
        set_lr(opt_g, LR_LATE)
        set_lr(opt_d, LR_LATE)
        print(f"\n[Epoch {epoch+1}] Switched to Phase-3 LR = {LR_LATE:.2e}")

    G.train(); D.train(); Extractor.train(); DiffFeat.train()

    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_psnr_c = 0.0
    epoch_psnr_s = 0.0
    n_batches    = 0

    # ── tqdm wraps the shorter of the two loaders ────────────────────────────
    pbar = tqdm(
        zip(cover_loader, wm_loader),
        total=min(len(cover_loader), len(wm_loader)),
        desc=f"Epoch {epoch+1}/{EPOCHS}",
        leave=False,
    )

    for cover, wm in pbar:          # ← fixed: no (img, label) tuple unpacking
        cover = cover.to(device)
        wm    = wm.to(device)

        # ════════════════════════════════════════════════════════════════════
        # A. Forward pass
        # ════════════════════════════════════════════════════════════════════
        watermarked = G(cover, wm)                          # embed

        # Apply random distortion attack  (Noiser picks one layer at random)
        attacked = attack_module([watermarked.clone(), cover.clone()])[0]

        # Differential features  (compensate for information lost in attack)
        diff_features = DiffFeat(watermarked, attacked)     # Xc - Xd features

        # Extract / recover secret image
        extracted_wm = Extractor(attacked, diff_features)   # recovered secret

        # ════════════════════════════════════════════════════════════════════
        # B. Train Discriminator  (update D first, per paper Section 3.2)
        # ════════════════════════════════════════════════════════════════════
        opt_d.zero_grad()

        d_real = D(cover)
        d_fake = D(attacked.detach())                       # detach: no G grad

        d_loss = discriminator_loss(d_real, d_fake)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
        opt_d.step()

        # ════════════════════════════════════════════════════════════════════
        # C. Train Generator + Extractor + DiffFeat
        # ════════════════════════════════════════════════════════════════════
        opt_g.zero_grad()

        # L_pre  — watermarked image should look like cover  (Eq. 1)
        L_pre = mse_loss(watermarked, cover)

        # L_post — extracted secret should look like original  (Eq. 2)
        L_post = mse_loss(extracted_wm, wm)

        # L_enhance — weighted combination  (Eq. 3)
        L_enhance = LAMBDA_C * L_pre + LAMBDA_S * L_post

        # L_adv — fool the discriminator  (Eq. 5/6)
        d_fake_for_g = D(attacked)
        L_adv = generator_adv_loss(d_fake_for_g)

        # L_f — differential feature loss  (Eq. 7)
        L_f = mse_loss(watermarked, attacked)

        # L_l — low-frequency wavelet loss  (Eq. 8)
        L_l = wavelet_ll_loss(watermarked, cover)

        # L_stage — total stage loss  (Eq. 9)
        g_loss = L_enhance + LAMBDA_ADV * L_adv + LAMBDA_F * L_f + L_l

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
        opt_g.step()

        # ════════════════════════════════════════════════════════════════════
        # D. Batch metrics (no_grad not needed — tensors already detached)
        # ════════════════════════════════════════════════════════════════════
        batch_psnr_c = psnr(watermarked, cover)
        batch_psnr_s = psnr(extracted_wm, wm)

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        epoch_psnr_c += batch_psnr_c
        epoch_psnr_s += batch_psnr_s
        n_batches    += 1

        pbar.set_postfix({
            "D": f"{d_loss.item():.3f}",
            "G": f"{g_loss.item():.3f}",
            "PSNR-C": f"{batch_psnr_c:.1f}",
            "PSNR-S": f"{batch_psnr_s:.1f}",
        })

    # ── Epoch-level averages ─────────────────────────────────────────────────
    avg_d    = epoch_d_loss / n_batches
    avg_g    = epoch_g_loss / n_batches
    avg_pc   = epoch_psnr_c / n_batches
    avg_ps   = epoch_psnr_s / n_batches

    print(
        f"Epoch [{epoch+1:>4}/{EPOCHS}] | "
        f"D: {avg_d:.4f} | G: {avg_g:.4f} | "
        f"PSNR-C: {avg_pc:.2f} dB | PSNR-S: {avg_ps:.2f} dB"
    )

    # ── Checkpoint every 100 epochs ─────────────────────────────────────────
    if (epoch + 1) % 100 == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1}.pth")
        torch.save({
            "epoch":         epoch + 1,
            "G":             G.state_dict(),
            "D":             D.state_dict(),
            "Extractor":     Extractor.state_dict(),
            "DiffFeat":      DiffFeat.state_dict(),
            "opt_g":         opt_g.state_dict(),
            "opt_d":         opt_d.state_dict(),
            "avg_psnr_c":    avg_pc,
            "avg_psnr_s":    avg_ps,
        }, ckpt_path)
        print(f"  ✓ Checkpoint saved → {ckpt_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. SAVE FINAL MODELS
# ─────────────────────────────────────────────────────────────────────────────
torch.save(G.state_dict(),         "generator_final.pth")
torch.save(D.state_dict(),         "discriminator_final.pth")  # ← was never saved before
torch.save(Extractor.state_dict(), "extractor_final.pth")
torch.save(DiffFeat.state_dict(),  "diff_extractor_final.pth")

print("\nTraining complete. All models saved.")