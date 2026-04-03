import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.generator import TextureSaliency

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cover_dir = "/kaggle/working/VKMA/data/DIV2K/cover"
    
    # Use os.walk to recursively search subfolders (like class0)
    img_path = None
    for root, dirs, files in os.walk(cover_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, file)
                break
        if img_path:
            break
            
    if not img_path:
        print(f"Could not find any images in {cover_dir} or its subdirectories.")
        return
        
    print(f"Using image: {img_path}")
    
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    cover_tensor = tf(img).unsqueeze(0).to(device)
    
    # Generate Saliency Mask
    saliency_module = TextureSaliency().to(device)
    with torch.no_grad():
        mask_tensor = saliency_module(cover_tensor)
    
    # Un-normalize for plotting
    cover_display = (cover_tensor[0].cpu().permute(1, 2, 0).numpy() * 0.5) + 0.5
    mask_display = mask_tensor[0, 0].cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cover_display)
    axes[0].set_title("Original Cover Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(mask_display, cmap='inferno')
    axes[1].set_title("Saliency-Guided Embedding Mask")
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig('saliency_demo.png', dpi=300)
    print("Saved saliency_demo.png successfully!")

if __name__ == "__main__":
    main()