import os
import torch
import matplotlib.pyplot as plt

def main():
    checkpoint_dir = "checkpoints"
    epochs = []
    psnr_c = []
    psnr_s = []

    for file_name in os.listdir(checkpoint_dir):
        if file_name.endswith(".pth"):
            file_path = os.path.join(checkpoint_dir, file_name)
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
            
            if "epoch" in checkpoint:
                epochs.append(checkpoint["epoch"])
                psnr_c.append(checkpoint["avg_psnr_c"])
                psnr_s.append(checkpoint["avg_psnr_s"])

    sorted_data = sorted(zip(epochs, psnr_c, psnr_s))
    epochs, psnr_c, psnr_s = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnr_c, label='PSNR-C (Cover)', color='blue', marker='o', linewidth=2)
    plt.plot(epochs, psnr_s, label='PSNR-S (Secret)', color='orange', marker='s', linewidth=2)
    
    plt.title('Watermarking PSNR Progression')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()