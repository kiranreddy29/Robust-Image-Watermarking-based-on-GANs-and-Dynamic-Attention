import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.generator import WatermarkGenerator
from models.discriminator import Discriminator
from noise_layers.noiser import Noiser
from noise_layers.Gaussian_noise import Gaussian_Noise
from noise_layers.identity import Identity

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")


class FlatImageDataset(Dataset):
    EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    def __init__(self, folder, transform=None):
        self.paths = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(self.EXTS):
                    self.paths.append(os.path.join(root, f))
        self.paths.sort()
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found in: {folder} or its subdirectories")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_loader(path, batch_size=4, shuffle=True):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = FlatImageDataset(path, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=(device == "cuda"), drop_last=True)


def psnr(img1, img2):
    mse = torch.mean((img1.detach() - img2.detach()) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(2.0 / math.sqrt(mse))


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

    noise_list    = [Gaussian_Noise(mean=0.0, sigma=GAUSSIAN_SIGMA), 'JpegPlaceholder', Identity()]
    attack_module = Noiser(noise_list, device=device).to(device)

    mse_loss = nn.MSELoss()

    opt_g = torch.optim.Adam(G.parameters(), lr=LR_EARLY)
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_EARLY)

    cover_loader = get_loader("data/DIV2K/cover",     batch_size=BATCH_SIZE, shuffle=True)
    wm_loader    = get_loader("data/DIV2K/watermark", batch_size=BATCH_SIZE, shuffle=True)

    print("Starting training ...")

    for epoch in range(EPOCHS):

        if epoch == PHASE3_EPOCH:
            for pg in opt_g.param_groups: pg['lr'] = LR_LATE
            for pg in opt_d.param_groups: pg['lr'] = LR_LATE
            print(f"\n[Epoch {epoch+1}] Phase 3 LR = {LR_LATE:.2e}")

        G.train()
        D.train()

        epoch_d_loss = 0.0
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
            cover = cover.to(device)
            wm    = wm.to(device)

            watermarked  = G(cover, wm)
            attacked     = attack_module([watermarked.clone(), cover.clone()])[0]
            extracted_wm = G.extract(attacked, watermarked)

            opt_d.zero_grad()
            d_real = D(cover)
            d_fake = D(attacked.detach())
            d_loss = discriminator_loss(d_real, d_fake)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_d.step()

            opt_g.zero_grad()

            L_pre     = mse_loss(watermarked, cover)
            L_post    = mse_loss(extracted_wm, wm)
            L_enhance = LAMBDA_C * L_pre + LAMBDA_S * L_post
            L_f       = mse_loss(watermarked, attacked)
            L_l       = wavelet_ll_loss(watermarked, cover)

            if epoch < PHASE2_EPOCH:
                g_loss = L_enhance + LAMBDA_F * L_f + L_l
            else:
                d_fake_for_g = D(attacked)
                L_adv  = generator_adv_loss(d_fake_for_g)
                g_loss = L_enhance + LAMBDA_ADV * L_adv + LAMBDA_F * L_f + L_l

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_g.step()

            batch_psnr_c  = psnr(watermarked, cover)
            batch_psnr_s  = psnr(extracted_wm, wm)
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

        avg_d  = epoch_d_loss / n_batches
        avg_g  = epoch_g_loss / n_batches
        avg_pc = epoch_psnr_c / n_batches
        avg_ps = epoch_psnr_s / n_batches

        print(
            f"Epoch [{epoch+1:>4}/{EPOCHS}] | "
            f"D: {avg_d:.4f} | G: {avg_g:.4f} | "
            f"PSNR-C: {avg_pc:.2f} dB | PSNR-S: {avg_ps:.2f} dB"
        )

        if (epoch + 1) % 100 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1}.pth")
            torch.save({
                "epoch":      epoch + 1,
                "G":          G.state_dict(),
                "D":          D.state_dict(),
                "opt_g":      opt_g.state_dict(),
                "opt_d":      opt_d.state_dict(),
                "avg_psnr_c": avg_pc,
                "avg_psnr_s": avg_ps,
            }, ckpt_path)
            print(f"  Checkpoint saved -> {ckpt_path}")

    torch.save(G.state_dict(), "generator_final.pth")
    torch.save(D.state_dict(), "discriminator_final.pth")

    print("\nTraining complete. Models saved.")

if __name__ == "__main__":
    main()