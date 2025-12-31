import copy
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader

from CNN.CNN import MultivariateCNN
from models_utils.Datasets import TrainDataframeWithLabels, pad_sequence
from models_utils.GLOBALS import *
from models_utils.utils import convert_to_features


def get_train_data(savename, modelType1, modelType2, target_size_type1, target_size_type2, embedding_size=128,
                   is_autoencoder=False):
    """
    :param savename: name of save
    :param modelType1: model for first type of sensor
    :param modelType2: model for second type of sensor
    :param target_size_type1: target size for first type files
    :param target_size_type2: target size for second type files
    :param embedding_size: size of embedding layer
    :param is_autoencoder: is autoencoder model
    :return: Dataframe with features and labels
    """

    data_type_1_list = []
    data_type_2_list = []
    train_data = pd.read_csv(TRAIN_CSV)  # Pre-read CSV file
    embedding_names = [f'embedding_feature_{i + 1}' for i in range(embedding_size)]

    for i, row in train_data.iterrows():
        class_path = os.path.join(files_directory, f"{row['id']}.csv")
        new_data = pd.read_csv(class_path)

        # ================== Type 1 ==================
        if new_data.shape[1] == 3:
            # get features from x,y,z data
            data_x_tensor = torch.tensor(new_data["x [m]"].values, dtype=torch.float32)
            data_y_tensor = torch.tensor(new_data["y [m]"].values, dtype=torch.float32)
            data_z_tensor = torch.tensor(new_data["z [m]"].values, dtype=torch.float32)
            new_features = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)

            # pad or cut data
            if len(new_data) < target_size_type1:
                new_data = pad_sequence(new_data, target_size_type1)
            else:
                new_data = new_data[:target_size_type1]

            # convert to tensor
            new_data = torch.tensor(new_data.values, dtype=torch.float32).to(device)

            # normalize
            normalized_new_data = (new_data - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)

            if not is_autoencoder:
                # transpose to [channels, seq_len] and add batch dimension [1, channels, seq_len]
                normalized_new_data = normalized_new_data.transpose(0, 1).unsqueeze(0)
                new_data_encoded = modelType1(normalized_new_data)
            else:
                if normalized_new_data.dim() == 2:
                    normalized_new_data = normalized_new_data.unsqueeze(0)  # (1, seq_len, features)

                new_data_encoded = modelType1.encode(normalized_new_data)


            model_features = new_data_encoded.squeeze(0).detach().cpu().numpy()

            # add file to results
            res = pd.DataFrame([model_features], columns=embedding_names)
            for col, value in new_features.items():
                res[col] = value
            res['activity'] = row['activity']
            data_type_1_list.append(res)

        # ================== Type 2 ==================
        else:
            new_data = new_data[new_data.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
            data_x_tensor = torch.tensor(new_data["x"].values, dtype=torch.float32)
            data_y_tensor = torch.tensor(new_data["y"].values, dtype=torch.float32)
            data_z_tensor = torch.tensor(new_data["z"].values, dtype=torch.float32)
            new_features = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)

            # pad or cut data
            if len(new_data) < target_size_type2:
                new_data = pad_sequence(new_data, target_size_type2)
            else:
                new_data = new_data[:target_size_type2]

            # convert to tensor
            new_data = torch.tensor(new_data.values, dtype=torch.float32).to(device)

            # normalize
            normalized_new_data = (new_data - min_values_type2) / (max_values_type2 - min_values_type2 + 1e-6)

            if not is_autoencoder:
                # transpose to [channels, seq_len] and add batch dimension [1, channels, seq_len]
                normalized_new_data = normalized_new_data.transpose(0, 1).unsqueeze(0)
                new_data_encoded = modelType2(normalized_new_data)
            else:
                if normalized_new_data.dim() == 2:
                    normalized_new_data = normalized_new_data.unsqueeze(0)  # (1, seq_len, features)

                new_data_encoded = modelType2.encode(normalized_new_data)

            model_features = new_data_encoded.squeeze(0).detach().cpu().numpy()

            # add file to results
            res = pd.DataFrame([model_features], columns=embedding_names)
            for col, value in new_features.items():
                res[col] = value
            res['activity'] = row['activity']
            data_type_2_list.append(res)

    # get all data
    data_type_1 = pd.concat(data_type_1_list, ignore_index=True)
    data_type_2 = pd.concat(data_type_2_list, ignore_index=True)
    data_type_1.to_csv(f'{savename}_type1.csv', index=False)
    data_type_2.to_csv(f'{savename}_type2.csv', index=False)
    return data_type_1, data_type_2



def train_cnn(save_name, data, data_type, data_size, num_epochs, batch_size=64, learning_rate=0.001):
    """
    Train a CNN model
    :param save_name: name of the model to save
    :param data: data to learn
    :param data_type: type of data (0,1) according to sensor
    :param data_size: size of data
    :param num_epochs: epochs to train for
    :param batch_size: batch size
    :param learning_rate: learning rate
    :return: trained CNN model
    """

    # get all data
    whole_dataset = TrainDataframeWithLabels(data, data_type, data_size)
    model_CNN = MultivariateCNN(3, data_size, 18).to(device)

    labels = data['activity'].to_list()
    train_idx, val_idx = train_test_split(
        range(len(whole_dataset)),
        test_size=0.2,  # 20% for validation
        stratify=labels,
    )

    # create stratified data
    train_dataset = Subset(whole_dataset, train_idx)
    val_dataset = Subset(whole_dataset, val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # training variables
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_CNN.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    best_val_loss = float('inf')  # Initialize best validation loss to infinity
    best_model_state = copy.deepcopy(model_CNN.state_dict())  # Initialize best model state

    for epoch in range(num_epochs):
        print("--------------")
        epoch_start_time = time.time()

        # batch variables
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        total_val_loss = 0
        val_correct = 0
        val_total = 0

        # training
        model_CNN.train()
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            normalized_inputs = inputs
            # normalized_inputs = (inputs - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
            normalized_inputs = normalized_inputs.transpose(1, 2)
            optimizer.zero_grad()
            outputs = model_CNN(normalized_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            total_train_loss += loss.item()

        # validation
        model_CNN.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                normalized_inputs = inputs
                # normalized_inputs = (inputs - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
                normalized_inputs = normalized_inputs.transpose(1, 2)
                outputs = model_CNN(normalized_inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                total_val_loss += loss.item()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # calculate metrics
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = total_train_loss / len(train_dataloader)
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        # check if this is the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model_CNN.state_dict())
            # save the best model to disk
            torch.save(best_model_state, f'{save_name}.pth')
            print(f'Saving Best Model with loss: {avg_val_loss}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Epoch Duration: {epoch_duration:.2f} seconds')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # load best model
    model_CNN = MultivariateCNN(3, data_size, 18).to(device)
    model_CNN.load_state_dict(torch.load(f'{save_name}.pth'))
    return model_CNN


def train_1D_cnn(save_name, data, data_type, data_size, num_epochs, batch_size=64, learning_rate=0.001):
    """
    Train a 1D CNN model
    """

    whole_dataset = TrainDataframeWithLabels(data, data_type, data_size)

    # Use 1 channel and flattened 3-axis data
    model_CNN = MultivariateCNN(1, data_size*3, 18).to(device)

    labels = data['activity'].to_list()
    train_idx, val_idx = train_test_split(
        range(len(whole_dataset)),
        test_size=0.2,
        stratify=labels,
    )

    train_dataset = Subset(whole_dataset, train_idx)
    val_dataset = Subset(whole_dataset, val_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_CNN.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model_CNN.state_dict())

    for epoch in range(num_epochs):
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        model_CNN.train()
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Flatten 3-axis data for 1D-CNN
            normalized_inputs = (inputs - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
            normalized_inputs = normalized_inputs.transpose(1, 2).reshape(inputs.shape[0], 1, 3*data_size)
            optimizer.zero_grad()
            outputs = model_CNN(normalized_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            total_train_loss += loss.item()

        # validation
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        model_CNN.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                normalized_inputs = (inputs - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
                normalized_inputs = normalized_inputs.transpose(1, 2).reshape(inputs.shape[0], 1, 3*data_size)
                outputs = model_CNN(normalized_inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model_CNN.state_dict())
            torch.save(best_model_state, f'{save_name}.pth')
            print(f'Saving Best Model with loss: {avg_val_loss}')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {100*train_correct/train_total:.2f}%, Validation Accuracy: {100*val_correct/val_total:.2f}%')

    # load the best model (same architecture!)
    model_CNN.load_state_dict(torch.load(f'{save_name}.pth'))
    return model_CNN
