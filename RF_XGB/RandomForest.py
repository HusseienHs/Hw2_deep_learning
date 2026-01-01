import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from models_utils.GLOBALS import files_directory  # kept import as-is (even if overwritten below)
from models_utils.utils import convert_to_features

# -----------------------
# Paths
# -----------------------
BASE_DIR = r"C:\Users\husseien\Desktop\340915149_322754953\Source Code"
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
files_directory = os.path.join(DATA_DIR, "unlabeled", "unlabeled")


def train_random_forest(data: pd.DataFrame, cols_to_drop, n_estimators: int):
    """
    Train a RandomForest model and print Accuracy + LogLoss.
    Returns: (model, label_encoder)
    """
    X = data.drop(columns=cols_to_drop)
    y = data["activity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train_enc)

    proba = model.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y_test_enc, pred)
    ll = log_loss(y_test_enc, proba)

    print("Accuracy:", acc)
    print("Log Loss:", ll)

    return model, le


def get_rf_data():
    """
    Build feature tables for RF/XGBoost from the per-id CSV files.
    Returns: (data_type_1_df, data_type_2_df)
    """
    t1_rows, t2_rows = [], []
    train_df = pd.read_csv(TRAIN_CSV)

    for _, row in train_df.iterrows():
        sample_id = row["id"]
        label = row["activity"]

        sample_path = os.path.join(files_directory, f"{sample_id}.csv")
        df = pd.read_csv(sample_path)

        # type 1: 4 columns (includes metric string column)
        if df.shape[1] == 4:
            df = df[df.iloc[:, 0] == "acceleration [m/s/s]"].iloc[:, 1:]
            x = torch.tensor(df["x"].values, dtype=torch.float32)
            y = torch.tensor(df["y"].values, dtype=torch.float32)
            z = torch.tensor(df["z"].values, dtype=torch.float32)

            feats = convert_to_features(x, y, z)
            feats["activity"] = label
            t1_rows.append(feats)
        else:
            x = torch.tensor(df["x [m]"].values, dtype=torch.float32)
            y = torch.tensor(df["y [m]"].values, dtype=torch.float32)
            z = torch.tensor(df["z [m]"].values, dtype=torch.float32)

            feats = convert_to_features(x, y, z)
            feats["activity"] = label
            t2_rows.append(feats)

    data_type_1 = pd.concat(t1_rows, ignore_index=True)
    data_type_2 = pd.concat(t2_rows, ignore_index=True)
    return data_type_1, data_type_2
