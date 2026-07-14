import torch
import torchvision.datasets
from torchvision import transforms

class MNISTSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.MNIST(
            root='~/.torch/datasets', 
            train=train, 
            download=True
        )
        self.X = self.dataset.data
        self.y = self.dataset.targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        image = self.X[index].to(torch.float32)
        image = (image / 127.5) - 1.0
        label = self.y[index].to(torch.long)
        return {'image': image, 'label': label}