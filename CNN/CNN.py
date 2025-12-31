from torch import nn
import torch.nn.functional as F


class MultivariateCNN(nn.Module):
    """
    Multivariate CNN class, last layer of CNN used as feature extractor.
    """
    def __init__(self, num_channels, input_length, num_classes=18):
        """
        Constructor
        :param num_channels: in our example it was 3: x,y,z
        :param input_length: length of each data, depends on sensor
        :param num_classes: last layer of feature extractor
        """
        super(MultivariateCNN, self).__init__()
        self.num_channels = num_channels
        self.input_length = input_length
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # calculate the size of the output from the last conv layer after pooling
        output_size_after_conv2 = input_length // 4

        self.fc1_size = 128 * output_size_after_conv2
        self.fc1 = nn.Linear(self.fc1_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward pass
        :param x: inputs of the model
        :return: outputs of the model
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_size)  # flatten the tensor for the fc layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
