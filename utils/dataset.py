from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loader(path, batch_size=1):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = ImageFolder(path, transform=tf)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)