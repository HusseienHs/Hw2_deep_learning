import os

import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Subset, DataLoader

from models_utils.Datasets import pad_sequence
from models_utils.GLOBALS import *
from NN.NeuralNetwork import NeuralNetwork
from models_utils.utils import convert_to_features


def get_train_data(ModelType1, ModelType2, embedding_size):
    """
    Get train data with all features
    :param ModelType1: first encoder model
    :param ModelType2: second encoder model
    :param embedding_size:
    :return:
    """
    data_type_1_list = []
    data_type_2_list = []
    train_data = pd.read_csv('train.csv')  # Pre-read CSV file
    embedding_names = [f'embedding_feature_{i + 1}' for i in range(embedding_size)]
    for i, row in train_data.iterrows():

        class_path = os.path.join(files_directory, f"{row['id']}.csv")
        new_data = pd.read_csv(class_path)

        # determine data type based on number of columns
        if new_data.shape[1] == 3:
            data_x_tensor = torch.tensor(new_data["x [m]"].values, dtype=torch.float32)
            data_y_tensor = torch.tensor(new_data["y [m]"].values, dtype=torch.float32)
            data_z_tensor = torch.tensor(new_data["z [m]"].values, dtype=torch.float32)
            new_features = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)

            if len(new_data) < 4000:
                new_data = pad_sequence(new_data, 4000)

            new_data = torch.tensor(new_data.values, dtype=torch.float32).to(device)
            new_data = new_data.view(1, new_data.shape[0], new_data.shape[1])
            normalized_new_data = (new_data - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
            new_data_encoded = ModelType1.encode(normalized_new_data)
            encoded_features = new_data_encoded.squeeze().detach().cpu().numpy()

            res = pd.DataFrame([encoded_features], columns=embedding_names)
            for col, value in new_features.items():
                res[col] = value
            res['activity'] = row['activity']

            data_type_1_list.append(res)
        else:
            new_data = new_data[new_data.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
            data_x_tensor = torch.tensor(new_data["x"].values, dtype=torch.float32)
            data_y_tensor = torch.tensor(new_data["y"].values, dtype=torch.float32)
            data_z_tensor = torch.tensor(new_data["z"].values, dtype=torch.float32)
            new_features = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)

            if len(new_data) < 1350:
                new_data = pad_sequence(new_data, 1350)

            new_data = torch.tensor(new_data.values, dtype=torch.float32).to(device)
            new_data = new_data.view(1, new_data.shape[0], new_data.shape[1])
            normalized_new_data = (new_data - min_values_type2) / (max_values_type2 - min_values_type2 + 1e-6)
            new_data_encoded = ModelType2.encode(normalized_new_data)
            encoded_features = new_data_encoded.squeeze().detach().cpu().numpy()

            res = pd.DataFrame([encoded_features], columns=embedding_names)
            for col, value in new_features.items():
                res[col] = value
            res['activity'] = row['activity']
            data_type_2_list.append(res)

    data_type_1 = pd.concat(data_type_1_list, ignore_index=True)
    data_type_2 = pd.concat(data_type_2_list, ignore_index=True)

    # save train data
    data_type_1.to_csv('train_data_type1.csv', index=False)
    data_type_2.to_csv('train_data_type2.csv', index=False)
    return data_type_1, data_type_2


def train_nn_model(dataset, data_type, input_size, hidden_sizes, num_classes, batch_size, learning_rate, num_epochs,
                   scheduler_factor, scheduler_patience):
    """
    Train neural network model
    :param dataset: data
    :param data_type: type of data (0,1) according to sensor
    :param input_size: size of the inputs
    :param hidden_sizes: size of the hidden layers
    :param num_classes: size of the outputs
    :param batch_size: batch size
    :param learning_rate: learning rate
    :param num_epochs: number of epochs
    :param scheduler_factor: factor for learning rate scheduler
    :param scheduler_patience: patience for learning rate scheduler
    :return:
    """
    model = NeuralNetwork(input_size, hidden_sizes, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience,
                                  verbose=True)

    labels = [y for _, y in dataset]
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        stratify=labels,
    )

    # create stratified data
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):

        total_train_loss = 0
        train_correct = 0
        train_total = 0

        total_val_loss = 0
        val_correct = 0
        val_total = 0

        # training
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            total_train_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                total_val_loss += loss.item()

        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = total_train_loss / len(train_dataloader)
        val_accuracy = 100 * val_correct / val_total
        avg_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(avg_val_loss)
        print("--------------")
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    # save model
    torch.save(model.state_dict(), f'Type{data_type}NNModel.pth')
    return model
