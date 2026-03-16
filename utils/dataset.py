from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_loader(path, batch_size=1, shuffle=False):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # ImageFolder will automatically look inside the 'class0' subdirectory
    dataset = ImageFolder(path, transform=tf)
    
    # num_workers=0 is safer for local testing to avoid multiprocessing crashes
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)