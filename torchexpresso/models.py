import torch


class NoopModel(torch.nn.Module):
    """
        Performs no actions. Directly returns the inputs.
    """

    def __init__(self):
        super(NoopModel, self).__init__()

    def forward(self, inputs, device):
        return inputs
