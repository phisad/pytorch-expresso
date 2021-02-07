import torch
from torch import nn


class FNNClassifier(nn.Module):

    def __init__(self, task):
        super(FNNClassifier, self).__init__()
        self.num_classes = task["num_classes"]
        self.image_shape = task["image_shape"]
        self.predictor = nn.Linear(in_features=self.image_shape[0] * self.image_shape[1] * self.image_shape[2],
                                   out_features=self.num_classes)

    def prepare(self, x, device):  # noqa
        x = torch.stack(x).to(device)  # stack listing and shift to gpu
        x = torch.flatten(x, start_dim=1)  # flatten starting from first image dim (ignore batch dim)
        return x

    def forward(self, x, device):
        return self.predictor(x)
