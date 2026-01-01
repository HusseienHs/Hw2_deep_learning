import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from models_utils.GLOBALS import files_directory


def pad_sequence(data: pd.DataFrame, max_sequence_length: int) -> pd.DataFrame:
    """
    Pad a sequence by appending reversed samples from the tail until reaching max length.
    """
    while len(data) < max_sequence_length:
        missing = max_sequence_length - len(data)
        pad_chunk = data[-missing:][::-1]
        data = pd.concat([data, pad_chunk], axis=0, ignore_index=True)

    return data[:max_sequence_length]


class DataframeWithLabels(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        x = torch.tensor(row.drop(labels=["activity"]).values, dtype=torch.float32)
        y = torch.tensor(row["activity"], dtype=torch.long)
        return x, y


class TrainDataframeWithLabels(Dataset):
    def __init__(self, dataframe: pd.DataFrame, data_type: str, max_sequence_length: int):
        self.data = dataframe
        self.data_type = data_type
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(files_directory, f'{row["id"]}.csv')
        df = pd.read_csv(path)

        if self.data_type == "2":
            df = df[df.iloc[:, 0] == "acceleration [m/s/s]"].iloc[:, 1:]

        if len(df) < self.max_sequence_length:
            df = pad_sequence(df, self.max_sequence_length)
        elif len(df) > self.max_sequence_length:
            df = df[:self.max_sequence_length]

        x = torch.tensor(df.values, dtype=torch.float32)
        y = torch.tensor(row["activity"], dtype=torch.long)
        return x, y


class TrainDataframeWithLabelsNoPad(Dataset):
    def __init__(self, dataframe: pd.DataFrame, data_type: str):
        self.data = dataframe
        self.data_type = data_type

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        path = os.path.join(files_directory, f'{row["id"]}.csv')
        df = pd.read_csv(path)

        if self.data_type == "2":
            df = df[df.iloc[:, 0] == "acceleration [m/s/s]"].iloc[:, 1:]

        x = torch.tensor(df.values, dtype=torch.float32)
        y = torch.tensor(row["activity"], dtype=torch.long)
        return x, y


class StandardDataset(Dataset):
    def __init__(self, files, max_sequence_length: int, data_type: str):
        self.data = files
        self.max_sequence_length = max_sequence_length
        self.data_type = data_type

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(files_directory, f"{self.data[idx]}.csv")
        df = pd.read_csv(path)

        if self.data_type == "2":
            df = df[df.iloc[:, 0] == "acceleration [m/s/s]"].iloc[:, 1:]

        if len(df) < self.max_sequence_length:
            df = pad_sequence(df, self.max_sequence_length)
        elif len(df) > self.max_sequence_length:
            df = df[:self.max_sequence_length]

        return torch.tensor(df.values, dtype=torch.float32)
