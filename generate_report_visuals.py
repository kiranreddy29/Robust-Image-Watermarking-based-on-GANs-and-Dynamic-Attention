import os
import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.generator import WatermarkGenerator

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
    parser = argparse.ArgumentParser(description="Generate visuals for VKMA model")
    parser.add_argument('--weights', type=str, default='generator_final.pth', 
                        help='Path to the trained generator weights file')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on device: {device}")

    # --- UPDATE THIS PATH ---
    WEIGHTS_PATH = args.weights  # <-- Update this to the correct path of your weights file
    
    G = WatermarkGenerator().to(device)
    try:
        G.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        G.eval()
        print(f"Successfully loaded weights from {WEIGHTS_PATH}!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    cover_path = get_first_image("data/DIV2K/cover")
    secret_path = get_first_image("data/DIV2K/watermark")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    cover_img = tf(Image.open(cover_path).convert('RGB')).unsqueeze(0).to(device)
    secret_img = tf(Image.open(secret_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        watermarked = G(cover_img, secret_img)
        extracted = G.extract(watermarked, watermarked)

    psnr_c = calc_psnr(watermarked, cover_img).item()
    psnr_s = calc_psnr(extracted, secret_img).item()
    
    print("-" * 30)
    print("FINAL EVALUATION METRICS:")
    print(f"Cover Imperceptibility (PSNR-C): {psnr_c:.2f} dB")
    print(f"Secret Recovery Quality (PSNR-S): {psnr_s:.2f} dB")
    print("-" * 30)

    def to_numpy(tensor):
        return (tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5) + 0.5

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(to_numpy(cover_img))
    axes[0].set_title("Original Cover")
    axes[0].axis('off')

    axes[1].imshow(to_numpy(watermarked))
    axes[1].set_title(f"Watermarked\nPSNR: {psnr_c:.2f} dB")
    axes[1].axis('off')

    axes[2].imshow(to_numpy(secret_img))
    axes[2].set_title("Original Secret")
    axes[2].axis('off')

    axes[3].imshow(to_numpy(extracted))
    axes[3].set_title(f"Extracted Secret\nPSNR: {psnr_s:.2f} dB")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('final_presentation_results.png', dpi=300)
    print("Saved final_presentation_results.png successfully!")

if __name__ == "__main__":
    main()