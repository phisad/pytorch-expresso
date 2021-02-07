import torch

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MnistDataset(Dataset):

    def __init__(self, task, split_name, params=None, device=None):
        self.samples = MNIST("mnist_data", train=True if split_name == "train" else False, download=True)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.to_tensor(sample[0])
        label = torch.as_tensor(sample[1], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.samples)
