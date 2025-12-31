import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder model for autoencoder
    """

    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_size = embedding_size
        self.hidden_size = 18
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        """
        Forward pass
        :param x: inputs - shape (batch, seq_len, input_size)
        :return: encoded outputs
        """
        x, (hidden_state, cell_state) = self.LSTM1(x)

        # handle batch size = 1
        if hidden_state.dim() == 3:
            # shape (num_layers, batch, hidden_size)
            last_hidden = hidden_state[-1, :, :]
        else:
            # shape (num_layers, hidden_size) -> add batch dimension
            last_hidden = hidden_state[-1, :].unsqueeze(0)

        return last_hidden



# (2) Decoder
class Decoder(nn.Module):
    """
    Decoder model for autoencoder
    """

    def __init__(self, seq_len, no_features, output_size):
        """
        :param seq_len: sequence length of the file
        :param no_features: number of features
        :param output_size: size of input
        """
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.hidden_size = (2 * no_features)
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass
        :param x: inputs - input, (h_0, c_0)
        :return: decoded outputs
        """
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))
        out = self.fc(x)
        return out


class LSTM_AE(nn.Module):
    """
    LSTM Autoencoder model
    """

    def __init__(self, seq_len, no_features, embedding_dim):
        """
        :param seq_len: sequence length of the file
        :param no_features: number of features
        :param embedding_dim: size of embedding layer
        """
        super().__init__()

        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features)

    def forward(self, x):
        """
        Forward pass
        :param x: inputs
        :return: encoded and decoded outputs
        """
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        """
        Encoding
        :param x: inputs
        :return: encoded inputs
        """
        self.eval()
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        """
        Decoding
        :param x: inputs
        :return: decoded inputs
        """
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded
