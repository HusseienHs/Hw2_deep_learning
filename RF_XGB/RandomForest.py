import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models_utils.GLOBALS import files_directory
from models_utils.utils import convert_to_features
import torch
import os

# Paths
BASE_DIR = r"C:\Users\husseien\Desktop\340915149_322754953\Source Code"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
files_directory = os.path.join(DATA_DIR, "unlabeled", "unlabeled")

def train_random_forest(data, cols_to_drop, n_estimators):
    """
    Trains a random forest model
    :param data: data to train / validate
    :param cols_to_drop: which columns to drop
    :param n_estimators: number of trees
    :return: rf model, label encoder
    """

    X = data.drop(columns=cols_to_drop)
    y = data['activity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # encode label
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # train rf
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_classifier.fit(X_train, y_train_encoded)

    # make predictions
    y_pred_proba = rf_classifier.predict_proba(X_test)
    y_pred_classes = np.argmax(y_pred_proba, axis=1)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    loss = log_loss(y_test, y_pred_proba)

    print("Accuracy:", accuracy)
    print("Log Loss:", loss)
    return rf_classifier, label_encoder


def get_rf_data():
    """
    Gets data for random forest / xgboost model
    :return: data
    """

    data_type_1_list = []
    data_type_2_list = []

    train_data = pd.read_csv(TRAIN_CSV)

    for i, row in train_data.iterrows():
        class_path = os.path.join(files_directory, f"{row['id']}.csv")
        new_data = pd.read_csv(class_path)

        # determine data type based on number of columns
        if new_data.shape[1] == 4:
            new_data = new_data[new_data.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
            data_x_tensor = torch.tensor(new_data["x"].values, dtype=torch.float32)
            data_y_tensor = torch.tensor(new_data["y"].values, dtype=torch.float32)
            data_z_tensor = torch.tensor(new_data["z"].values, dtype=torch.float32)
            new_data = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)
            new_data['activity'] = row['activity']
            data_type_1_list.append(new_data)
        else:
            data_x_tensor = torch.tensor(new_data["x [m]"].values, dtype=torch.float32)
            data_y_tensor = torch.tensor(new_data["y [m]"].values, dtype=torch.float32)
            data_z_tensor = torch.tensor(new_data["z [m]"].values, dtype=torch.float32)
            new_data = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)
            new_data['activity'] = row['activity']
            data_type_2_list.append(new_data)

    data_type_1 = pd.concat(data_type_1_list, ignore_index=True)
    data_type_2 = pd.concat(data_type_2_list, ignore_index=True)
    return data_type_1, data_type_2
