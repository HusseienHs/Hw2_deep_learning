import torch
import torch.nn as nn


class Encoder(nn.Module):
    """LSTM-based encoder for an autoencoder."""

    def __init__(self, seq_len: int, no_features: int, embedding_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_size = embedding_size

        # kept exactly as in your code (even if unused elsewhere)
        self.hidden_size = 18

        self.lstm = nn.LSTM(
            input_size=no_features,
            hidden_size=embedding_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        """x: (batch, seq_len, input_size) -> returns last hidden state per batch."""
        _, (h_n, _) = self.lstm(x)

        # handle batch size = 1 (preserve same behavior)
        last_h = h_n[-1] if h_n.dim() != 3 else h_n[-1, :, :]
        if last_h.dim() == 1:
            last_h = last_h.unsqueeze(0)

        return last_h


class Decoder(nn.Module):
    """LSTM-based decoder for an autoencoder."""

    def __init__(self, seq_len: int, no_features: int, output_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.output_size = output_size

        self.hidden_size = 2 * no_features

        self.lstm = nn.LSTM(
            input_size=no_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.proj = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch, embedding_dim)
        repeat across time -> LSTM -> Linear -> (batch, seq_len, output_size)
        """
        x_rep = x.unsqueeze(1).expand(-1, self.seq_len, -1)
        y, _ = self.lstm(x_rep)
        y = y.reshape(-1, self.seq_len, self.hidden_size)
        return self.proj(y)


class LSTM_AE(nn.Module):
    """LSTM Autoencoder model."""

    def __init__(self, seq_len: int, no_features: int, embedding_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(seq_len, no_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, no_features)

    def forward(self, x):
        torch.manual_seed(0)  # preserve original behavior exactly
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def encode(self, x):
        self.eval()
        return self.encoder(x)

    def decode(self, x):
        self.eval()
        x_hat = self.decoder(x)
        return x_hat.squeeze()
