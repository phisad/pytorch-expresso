import torch
from torch import nn


class FNNClassifier(nn.Module):

    def __init__(self, task, hidden_size=512):
        super(FNNClassifier, self).__init__()
        self.num_classes = task["num_classes"]
        self.image_shape = task["image_shape"]
        self.image_embedding = nn.Linear(in_features=self.image_shape[0] * self.image_shape[1] * self.image_shape[2],
                                         out_features=hidden_size)
        self.predictor = nn.Linear(in_features=hidden_size, out_features=self.num_classes)

    def prepare(self, x, device):  # noqa
        x = torch.stack(x).to(device)  # stack listing and shift to gpu
        x = torch.flatten(x, start_dim=1)  # flatten starting from first image dim (ignore batch dim)
        return x

    def forward(self, x, device):
        image_embedding = torch.relu(self.image_embedding(x))
        return self.predictor(image_embedding)


class Conv2dBlock(nn.Module):

    def __init__(self, channels, features, kernels, padding):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernels, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # The standard non-linearity for convolutional neural networks
        self.relu = nn.ReLU()
        # Store indicies for deconvolution
        self.indicies = []

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x, indicies = self.pool(x)
        self.indicies = indicies
        return x


class CNNClassifier(nn.Module):

    def __init__(self, task, params=None):
        super(CNNClassifier, self).__init__()
        self.num_classes = task["num_classes"]
        self.image_shape = task["image_shape"]
        self.block1 = Conv2dBlock(self.image_shape[0], 8, 3, 0)
        self.block2 = Conv2dBlock(8, 16, 3, 0)
        self.block3 = Conv2dBlock(16, 32, 3, 0)
        self.predictor = nn.Linear(in_features=32, out_features=self.num_classes)

    def prepare(self, x, device):  # noqa
        x = torch.stack(x).to(device)  # stack listing and shift to gpu
        return x

    def forward(self, x, device):
        # 32 x  1 x 28 x 28
        x = self.block1(x)
        # 32 x  8 x 13 x 13
        x = self.block2(x)
        # 32 x 32 x  1 x  1
        x = self.block3(x)
        image_embedding = torch.squeeze(x)
        return self.predictor(image_embedding)
