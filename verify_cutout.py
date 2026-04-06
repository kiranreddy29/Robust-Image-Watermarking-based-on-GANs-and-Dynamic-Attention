import os
import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from models.generator import WatermarkGenerator
from noise_layers.cutout import CutoutAttack

def calc_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

def get_first_image(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                return os.path.join(root, file)
    return None

def main():
    parser = argparse.ArgumentParser(description="Verify Cutout Attack")
    parser.add_argument('--weights', type=str, default='generator_final.pth', 
                        help='Path to the trained generator weights file')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Verifying Cutout Attack on: {device}")

    WEIGHTS_PATH = args.weights
    G = WatermarkGenerator().to(device)
    
    try:
        G.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        G.eval()
        print(f"Successfully loaded {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Using the Kaggle paths you confirmed earlier
    cover_path = get_first_image("data/DIV2K/cover")
    secret_path = get_first_image("data/DIV2K/watermark")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    cover_img = tf(Image.open(cover_path).convert('RGB')).unsqueeze(0).to(device)
    secret_img = tf(Image.open(secret_path).convert('RGB')).unsqueeze(0).to(device)

    # Embed the Secret
    with torch.no_grad():
        watermarked = G.embed(cover_img, secret_img)
        
        # MANUALLY APPLY THE CUTOUT ATTACK (Drop 15% of data)
        attacker = CutoutAttack(drop_prob=0.15, block_size=48).to(device)
        attacked_watermarked = attacker([watermarked.clone(), cover_img.clone()])[0]
        
        # Extract from the Damaged Image
        extracted = G.extract(attacked_watermarked, watermarked)

    psnr_c = calc_psnr(watermarked, cover_img).item()
    psnr_s = calc_psnr(extracted, secret_img).item()

    print("-" * 30)
    print("CUTOUT SURVIVAL METRICS:")
    print(f"Cover PSNR (Before Attack): {psnr_c:.2f} dB")
    print(f"Extracted PSNR (After Attack): {psnr_s:.2f} dB")
    print("-" * 30)

    # Generate the Slide Visuals
    def to_numpy(tensor):
        return (tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5) + 0.5

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(to_numpy(cover_img))
    axes[0].set_title("Original Cover")
    axes[0].axis('off')

    axes[1].imshow(to_numpy(attacked_watermarked))
    axes[1].set_title("Vandalized Watermarked\n(15% Data Destroyed)")
    axes[1].axis('off')

    axes[2].imshow(to_numpy(secret_img))
    axes[2].set_title("Original Secret")
    axes[2].axis('off')

    axes[3].imshow(to_numpy(extracted))
    axes[3].set_title(f"Extracted Survivor\nPSNR: {psnr_s:.2f} dB")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('cutout_verification_slide.png', dpi=300)
    print("Saved cutout_verification_slide.png! Ready for PowerPoint.")

if __name__ == "__main__":
    main()