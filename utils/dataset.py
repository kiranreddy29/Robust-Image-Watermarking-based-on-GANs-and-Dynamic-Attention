import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class FlatImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.paths = []
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.paths.append(os.path.join(root, file))
        
        self.paths.sort()
        
        if len(self.paths) == 0:
            raise RuntimeError(f"Found 0 images in {folder} or its subdirectories.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img

def get_loader(path, batch_size=1, shuffle=False):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    return DataLoader(FlatImageDataset(path, transform=tf), batch_size=batch_size, shuffle=shuffle, num_workers=0)