import torch.nn as nn


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1):
        super().__init__()
        padding = dilation
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0,)

    def forward(self, x):
        x_orig = x
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True):
        super().__init__()

        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth) for depth in range(n_depth)]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
