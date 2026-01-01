from torch import nn
import torch.nn.functional as F


class MultivariateCNN(nn.Module):
    """1D CNN for multivariate time-series (final layer can act as feature extractor)."""

    def __init__(self, num_channels, input_length, num_classes=18):
        super().__init__()
        self.num_channels = num_channels
        self.input_length = input_length
        self.num_classes = num_classes

        self.block1 = nn.Conv1d(
            in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.block2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.down = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # after two poolings: length becomes input_length // 4
        conv_len = input_length // 4
        self.flat_dim = 128 * conv_len

        self.head1 = nn.Linear(self.flat_dim, 512)
        self.head2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.down(F.relu(self.block1(x)))
        x = self.down(F.relu(self.block2(x)))

        x = x.reshape(-1, self.flat_dim)
        x = F.relu(self.head1(x))
        return self.head2(x)
