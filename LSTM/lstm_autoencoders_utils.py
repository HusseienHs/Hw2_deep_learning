from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models_utils.GLOBALS import *
from models_utils.Datasets import StandardDataset
from LSTM.lstm_autoencoder import LSTM_AE


def train_lstm_autoencoder(data, data_type, target_size, embedding_size, learning_rate, batch_size, num_epochs):
    """
    Train an LSTM autoencoder.
    Args:
        data: data to learn
        data_type: type of data ('1' or otherwise) according to sensor
        target_size: target size of the data
        embedding_size: embedding dimension
        learning_rate: optimizer lr
        batch_size: batch size
        num_epochs: number of epochs
    Returns:
        trained LSTM autoencoder
    """
    model = LSTM_AE(target_size, 3, embedding_size).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss(reduction="mean")

    lr_sched = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=0)

    dataset = StandardDataset(data, target_size, data_type)

    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    ds_train, ds_val = random_split(dataset, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    def _normalize(tensor_batch):
        if data_type == "1":
            mn, mx = min_values_type1, max_values_type1
        else:
            mn, mx = min_values_type2, max_values_type2
        return (tensor_batch - mn) / (mx - mn + 1e-6)

    for ep in range(num_epochs):
        model.train()
        running_train = 0.0

        for i, xb in enumerate(dl_train, start=1):
            xb = xb.to(device)
            xb = _normalize(xb)

            _, recon = model(xb)
            loss = mse(recon, xb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_train += loss.item()
            if (i % 20 == 0) or (i == len(dl_train)):
                print(f"Batch: {i}/{len(dl_train)}, Train Loss: {loss.item():.4f}")

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb in dl_val:
                xb = xb.to(device)
                xb = _normalize(xb)

                _, recon = model(xb)
                running_val += mse(recon, xb).item()

        avg_train = running_train / len(dl_train)
        avg_val = running_val / len(dl_val)

        lr_sched.step(avg_val)
        print(
            f"Epoch [{ep + 1}/{num_epochs}], "
            f"Average Training Loss: {avg_train:.4f}, , "
            f"Average Validation Loss: {avg_val:.4f}"
        )

    torch.save(model.state_dict(), f"Type{data_type}LSTMAutoencoder.pth")
    return model
