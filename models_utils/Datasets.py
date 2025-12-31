import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from models_utils.GLOBALS import files_directory


def pad_sequence(data, max_sequence_length):
    """
    Pads a sequence of data with inverse data from the last timestamp
    :param data: data to pad
    :param max_sequence_length: maximum sequence length
    :return:
    """
    while len(data) < max_sequence_length:
        pad_size = max_sequence_length - len(data)
        pad_values = data[-pad_size:][::-1]
        data = pd.concat([data, pad_values], axis=0, ignore_index=True)
    return data[:max_sequence_length]


class DataframeWithLabels(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        x = torch.tensor(item.drop(labels=['activity']).values, dtype=torch.float32)
        y = torch.tensor(item['activity'], dtype=torch.long)
        return x, y


class TrainDataframeWithLabels(Dataset):
    def __init__(self, dataframe, data_type, max_sequence_length):
        self.data = dataframe
        self.data_type = data_type
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        file_path = os.path.join(files_directory, f'{item["id"]}.csv', )
        x = pd.read_csv(file_path)
        if self.data_type == '2':
            x = x[x.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
        if len(x) < self.max_sequence_length:
            x = pad_sequence(x, self.max_sequence_length)
        elif len(x) > self.max_sequence_length:
            x = x[:self.max_sequence_length]

        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(item['activity'], dtype=torch.long)
        return x, y


class TrainDataframeWithLabelsNoPad(Dataset):
    def __init__(self, dataframe, data_type):
        self.data = dataframe
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        file_path = os.path.join(files_directory, f'{item["id"]}.csv', )
        x = pd.read_csv(file_path)
        if self.data_type == '2':
            x = x[x.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(item['activity'], dtype=torch.long)
        return x, y


class StandardDataset(Dataset):
    def __init__(self, files, max_sequence_length, data_type):
        self.data = files
        self.max_sequence_length = max_sequence_length
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = os.path.join(files_directory, f'{self.data[idx]}.csv', )
        x = pd.read_csv(file_path)
        if self.data_type == '2':
            x = x[x.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
        if len(x) < self.max_sequence_length:
            x = pad_sequence(x, self.max_sequence_length)
        elif len(x) > self.max_sequence_length:
            x = x[:self.max_sequence_length]
        x = torch.tensor(x.values, dtype=torch.float32)
        return x
