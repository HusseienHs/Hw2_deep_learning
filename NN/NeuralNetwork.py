from torch import nn


class NeuralNetwork(nn.Module):
    """
    Neural Network model with flexible layers
    """
    def __init__(self, input_size, hidden_sizes, num_classes):
        """
        Constructor
        :param input_size: size of the input
        :param hidden_sizes: sizes of the hidden layers
        :param num_classes: size of the output layer
        """
        super(NeuralNetwork, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_sizes[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass
        :param x: inputs
        :return: outputs
        """
        out = self.hidden_layers(x)
        out = self.fc_out(out)
        out = self.softmax(out)
        return out
