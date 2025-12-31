from torch import optim, nn
from torch.utils.data import random_split, DataLoader
from models_utils.GLOBALS import *
from models_utils.Datasets import StandardDataset
from LSTM.lstm_autoencoder import LSTM_AE
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_lstm_autoencoder(data, data_type, target_size, embedding_size, learning_rate, batch_size, num_epochs):
    """
    Train LSTM autoencoder
    :param data: data to learn
    :param data_type: type of data (0,1) according to sensor
    :param target_size: target size of the data
    :param embedding_size: size of embedding layer
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param num_epochs: epochs to train for
    :return: trained LSTM autoencoder
    """

    LSTMAutoencoder = LSTM_AE(target_size, 3, embedding_size).to(device)
    optimizer = optim.Adam(LSTMAutoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    whole_dataset = StandardDataset(data, target_size, data_type)

    train_size = int(0.8 * len(whole_dataset))
    val_size = len(whole_dataset) - train_size
    train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        LSTMAutoencoder.train()
        total_train_loss = 0

        # training
        for batch_idx, batch_data in enumerate(train_dataloader):
            batch_data = batch_data.to(device)

            # normalize batch data based on its own min and max for each feature
            if data_type == '1':
                normalized_batch_data = (batch_data - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
            else:
                normalized_batch_data = (batch_data - min_values_type2) / (max_values_type2 - min_values_type2 + 1e-6)

            embedding, outputs = LSTMAutoencoder(normalized_batch_data)

            loss = criterion(outputs, normalized_batch_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(train_dataloader):
                print(f'Batch: {batch_idx + 1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f}')

        # validation
        LSTMAutoencoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_data in val_dataloader:
                batch_data = batch_data.to(device)
                # normalize validation data
                if data_type == '1':
                    normalized_batch_data = (batch_data - min_values_type1) / (
                            max_values_type1 - min_values_type1 + 1e-6)
                else:
                    normalized_batch_data = (batch_data - min_values_type2) / (
                            max_values_type2 - min_values_type2 + 1e-6)

                embedding, outputs = LSTMAutoencoder(normalized_batch_data)
                loss = criterion(outputs, normalized_batch_data)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}, , Average Validation Loss: {avg_val_loss:.4f}')
    torch.save(LSTMAutoencoder.state_dict(), f'Type{data_type}LSTMAutoencoder.pth')
    return LSTMAutoencoder
