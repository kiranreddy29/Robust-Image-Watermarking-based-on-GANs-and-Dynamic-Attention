# Robust Image Watermarking — GAN + Dynamic Attention

Implementation of the paper:
**"Robust Image Watermarking Algorithm Based on Generative Adversarial Networks and Dynamic Attention"**
Yan et al., *Neurocomputing*, 2026.

---

## Overview

This project implements a three-layer deep watermarking framework that embeds a secret image invisibly into a cover image and recovers it robustly after distortion attacks.

| Layer | Component | Purpose |
|---|---|---|
| Perception | GAN (Generator + Steganalysis Discriminator) | Makes watermarked images statistically indistinguishable from natural images |
| Resistance | Dynamic Attention Enhancement (Swin Transformer) | Handles Gaussian noise (window=4) and JPEG artifacts (window=8) |
| Recovery | Differential Feature Extractor (4-layer DenseNet) | Compensates for information lost during attacks |

---

## Architecture

```
EMBEDDING
  cover image (xh)  ──┐
                       ├── [cat 6ch] ── InvertibleBlock (ISN) ── [ch 0:3] ── watermarked (xc)
  secret image (xs) ──┘

EXTRACTION
  watermarked (xc) ──┐
                      ├── DifferentialFeatureExtractor ── xc_feat
  attacked (xd)    ──┘
  attacked (xd) ────── EnhancementModule (pre, window=4) ── xd_enhanced
  [xd_enhanced + xc_feat] ── InvertibleBlock reverse ── [ch 3:6] ── EnhancementModule (post, window=8) ── extracted (xe)
```

---

## Project Structure

```
├── train.py                  Main training script
├── test.py                   Evaluation on full test set (PSNR-C avg)
├── test_basic.py             Quick single-batch pipeline test
├── plot_metrics.py           Plot PSNR-C / PSNR-S from checkpoints
├── requirements.txt
├── models/
│   ├── generator.py          WatermarkGenerator — ISN, Enhancement, DiffFeat
│   └── discriminator.py      Steganalysis discriminator
├── noise_layers/
│   ├── Gaussian_noise.py     Gaussian noise attack
│   ├── jpeg_compression.py   Differentiable JPEG simulation
│   ├── quantization.py       Round-error simulation
│   ├── identity.py           No-op (pass-through)
│   └── noiser.py             Randomly picks one attack per batch
└── utils/
    ├── dataset.py            Flat-folder image loader (no class subfolders needed)
    └── metrics.py            PSNR utility
```

---

## Dataset Setup

This project uses the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/).

```
data/
└── DIV2K/
    ├── cover/       ← carrier images (e.g. DIV2K train HR, 800 images)
    └── watermark/   ← secret images  (e.g. DIV2K val HR,  100 images)
```

Any flat folder of `.png` / `.jpg` images works — no class subdirectories required. For quick local testing you can use any small set of images.

---

## Installation

**Requirements:** Python 3.9+, pip

```bash
git clone https://github.com/kiranreddy29/robust-image-watermarking-based-on-gans-and-dynamic-attention.git
cd robust-image-watermarking-based-on-gans-and-dynamic-attention

pip install -r requirements.txt
```

`requirements.txt` contents:
```
torch>=2.0.0
torchvision>=0.15.0
numpy
matplotlib
Pillow
timm
kornia
tqdm
```

---

## Training

```bash
python train.py
```

**What happens:**
- **Phase 1** (epochs 0–299): trains ISN + Enhancement modules with image quality loss only — no adversarial loss yet, ensures stable base quality
- **Phase 2** (epochs 300–999): adversarial loss introduced, discriminator trains against generator to improve watermark concealment
- **Phase 3** (epochs 1000–1600): learning rate drops from `10⁻⁴·⁵` to `10⁻⁵·⁵` for fine-tuning

Checkpoints are saved every 100 epochs to `checkpoints/ckpt_epochN.pth`.

Final models saved as:
- `generator_final.pth`
- `discriminator_final.pth`

**Key hyperparameters** (all in `train.py`):

| Parameter | Value | Paper reference |
|---|---|---|
| `EPOCHS` | 1600 | Section 4.1 |
| `BATCH_SIZE` | 4 (reduce if OOM) | 16 in paper |
| `LR_EARLY` | 10⁻⁴·⁵ | Section 4.1 |
| `LR_LATE` | 10⁻⁵·⁵ | Section 4.1 |
| `LAMBDA_ADV` | 0.1 | Eq. 9 |
| `LAMBDA_F` | 0.05 | Eq. 9 |
| `LAMBDA_C` | 1.0 | Section 3.5 |
| `LAMBDA_S` | 1.0 | Section 3.5 |

---

## Testing

**Quick pipeline test** (single batch, checks all 4 steps work):
```bash
python test_basic.py
```
Output:
```
PSNR-C (Cover vs Watermarked): XX.XX dB
PSNR-S (Secret vs Extracted):  XX.XX dB
Pipeline test completed successfully!
```

**Full evaluation** (averages PSNR-C over entire test set):
```bash
python test.py
```

**Plot training curves** (requires at least one checkpoint):
```bash
python plot_metrics.py
```
Saves `training_metrics.png` showing PSNR-C and PSNR-S over epochs.

---

## Metrics

| Metric | Description | Target |
|---|---|---|
| **PSNR-C** | Peak Signal-to-Noise Ratio between cover and watermarked image | Higher = more invisible watermark |
| **PSNR-S** | Peak Signal-to-Noise Ratio between secret and extracted image | Higher = better recovery after attack |

Paper results on DIV2K (Level 1, Gaussian σ=10): PSNR-C = 36.87 dB, PSNR-S = 34.31 dB.

---

## Attacks Tested

| Attack | Implementation | Paper setting |
|---|---|---|
| Gaussian noise | `Gaussian_noise.py` | σ = 1, σ = 10 |
| JPEG compression | `jpeg_compression.py` | QF = 90, QF = 80 |
| Round error | `quantization.py` | float→int rounding |
| None (baseline) | `identity.py` | — |

During training, one attack is randomly selected per batch via `noiser.py`.

---

## Team

Group 16 — B.Tech CSE

| Roll No | Name |
|---|---|
| 23BCS275 | Yetra Rama Kiran Aravind Reddy|
| 23BCS257 | Vasantha Venkateswara Rao |
| 23BCS026 | Anchal Siddharth Patil |
| 23BCS052 | Banoth Meenakshi |

---

## Reference

Yan J, Tian L, Xu X, Li C — *Robust Image Watermarking Algorithm Based on Generative Adversarial Networks and Dynamic Attention* — Neurocomputing, 2026.
DOI: https://doi.org/10.1016/j.neucom.2026.132829

## Hardware & Reproducibility
- **Recommended GPU:** 16GB+ VRAM (e.g., Tesla P100, RTX 4080) for batch size 16. Use batch size 4 for 8GB VRAM.
- **Training Time:** ~12 hours on Kaggle P100 GPU for 1600 epochs.
- **Reproducibility:** PyTorch random seeds were fixed during testing to ensure consistent attack evaluation.

## License
This project is licensed under the MIT License.