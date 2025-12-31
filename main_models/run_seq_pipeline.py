
# run_seq_pipeline.py
# ------------------------------------------------------------
# End-to-end pipeline:
# 1) Extract per-sample features: CNN logits (18-d) + engineered stats
# 2) Build a pseudo-labeled temporal sequence by filling gaps where
#    the label is the same on both sides of a gap (strict leak-exploit)
# 3) Train an LSTM to predict the NEXT label from the last seq_len steps
# 4) Predict probabilities for the competition test ids (sample_submission.csv)
#
# This script is designed to be "drop-in runnable" on your machine.
# You only need to set --data_dir to the folder that contains:
#   data/train.csv
#   data/sample_submission.csv
#   data/unlabeled/unlabeled/<id>.csv
#
# Example:
#   python run_seq_pipeline.py --data_dir "C:\...\Source Code\data" --cutoff_id 77000
#
# Outputs:
#   - features_cache/features_<cutoff_id>.parquet (optional cache)
#   - seq_lstm.pth (trained model)
#   - submission_seq_lstm.csv (submission file)
# ------------------------------------------------------------

import argparse
import os
from pathlib import Path
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Config: min/max normalizers
# (copied from your GLOBALS.py)
# -----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def min_max_tensors(device):
    min_values_type1 = torch.tensor([-2.30336538209, -3.40176282264, -0.976451779318], device=device)
    max_values_type1 = torch.tensor([ 7.2808711902 ,  3.9957190444 ,  4.76762666942 ], device=device)
    min_values_type2 = torch.tensor([-19.603912, -19.594337, -19.603912], device=device)
    max_values_type2 = torch.tensor([ 19.594337,  19.603912,  19.594337], device=device)
    return min_values_type1, max_values_type1, min_values_type2, max_values_type2


# -----------------------------
# Activity mapping (from sample_submission columns)
# -----------------------------
def activity_mapping_from_submission(sub_df: pd.DataFrame):
    # sub columns: sample_id, then classes
    classes = [c for c in sub_df.columns if c != "sample_id"]
    activity_id_mapping = {label: i for i, label in enumerate(classes)}
    id_activity_mapping = {i: label for label, i in activity_id_mapping.items()}
    return classes, activity_id_mapping, id_activity_mapping


# -----------------------------
# CNN Feature Extractor
# Your CNN produces logits of size num_classes=18.
# We'll use logits as an "embedding" (works well in your repo too).
# -----------------------------
class MultivariateCNN(nn.Module):
    def __init__(self, num_channels, input_length, num_classes=18):
        super().__init__()
        self.num_channels = num_channels
        self.input_length = input_length
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        output_size_after_conv2 = input_length // 4
        self.fc1_size = 128 * output_size_after_conv2
        self.fc1 = nn.Linear(self.fc1_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, C, L)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.fc1_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Padding helper (copied from your Datasets.py)
# -----------------------------
def pad_sequence_df(data: pd.DataFrame, max_sequence_length: int) -> pd.DataFrame:
    while len(data) < max_sequence_length:
        pad_size = max_sequence_length - len(data)
        pad_values = data[-pad_size:][::-1]
        data = pd.concat([data, pad_values], axis=0, ignore_index=True)
    return data[:max_sequence_length]


# -----------------------------
# Feature engineering (ported from your utils.py)
# Returns 1-row DataFrame with many stats.
# -----------------------------
def convert_to_features(data_x: torch.Tensor, data_y: torch.Tensor, data_z: torch.Tensor) -> pd.DataFrame:
    def manual_skewness(data):
        n = len(data)
        mean = torch.mean(data)
        std_dev = torch.std(data, unbiased=True)
        return (n / ((n - 1) * (n - 2))) * torch.sum(((data - mean) / (std_dev + 1e-9)) ** 3)

    def manual_kurtosis(data):
        n = len(data)
        mean = torch.mean(data)
        std_dev = torch.std(data, unbiased=True)
        return (n * (n + 1) * torch.sum(((data - mean) / (std_dev + 1e-9)) ** 4) / ((n - 1) * (n - 2) * (n - 3))) - (
            3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )

    mean_x, mean_y, mean_z = torch.mean(data_x), torch.mean(data_y), torch.mean(data_z)
    std_x, std_y, std_z = torch.std(data_x), torch.std(data_y), torch.std(data_z)
    med_x, med_y, med_z = torch.median(data_x), torch.median(data_y), torch.median(data_z)

    between_x = (torch.max(data_x) - torch.min(data_x)) / max(len(data_x), 1)
    between_y = (torch.max(data_y) - torch.min(data_y)) / max(len(data_y), 1)
    between_z = (torch.max(data_z) - torch.min(data_z)) / max(len(data_z), 1)

    sum_x, sum_y, sum_z = torch.sum(data_x), torch.sum(data_y), torch.sum(data_z)

    sma_x = torch.mean(torch.abs(data_x - torch.mean(data_x)))
    sma_y = torch.mean(torch.abs(data_y - torch.mean(data_y)))
    sma_z = torch.mean(torch.abs(data_z - torch.mean(data_z)))

    count_x, count_y, count_z = len(data_x), len(data_y), len(data_z)

    p2p_x = torch.max(data_x) - torch.min(data_x)
    p2p_y = torch.max(data_y) - torch.min(data_y)
    p2p_z = torch.max(data_z) - torch.min(data_z)

    skew_x, skew_y, skew_z = manual_skewness(data_x), manual_skewness(data_y), manual_skewness(data_z)
    kurt_x, kurt_y, kurt_z = manual_kurtosis(data_x), manual_kurtosis(data_y), manual_kurtosis(data_z)

    rms_x = torch.sqrt(torch.mean(data_x ** 2))
    rms_y = torch.sqrt(torch.mean(data_y ** 2))
    rms_z = torch.sqrt(torch.mean(data_z ** 2))

    zcr_x = ((data_x[:-1] * data_x[1:]) < 0).sum() if len(data_x) > 1 else torch.tensor(0.0)
    zcr_y = ((data_y[:-1] * data_y[1:]) < 0).sum() if len(data_y) > 1 else torch.tensor(0.0)
    zcr_z = ((data_z[:-1] * data_z[1:]) < 0).sum() if len(data_z) > 1 else torch.tensor(0.0)

    sma_global = torch.mean(torch.abs(data_x) + torch.abs(data_y) + torch.abs(data_z))

    max_ix_x, max_ix_y, max_ix_z = torch.argmax(data_x), torch.argmax(data_y), torch.argmax(data_z)
    min_ix_x, min_ix_y, min_ix_z = torch.argmin(data_x), torch.argmin(data_y), torch.argmin(data_z)

    fft_x = torch.fft.fft(data_x)
    fft_y = torch.fft.fft(data_y)
    fft_z = torch.fft.fft(data_z)

    dom_fx = torch.argmax(torch.abs(fft_x))
    dom_fy = torch.argmax(torch.abs(fft_y))
    dom_fz = torch.argmax(torch.abs(fft_z))

    data_dic = {
        'mean_x': mean_x.item(), 'mean_y': mean_y.item(), 'mean_z': mean_z.item(),
        'std_deviation_x': std_x.item(), 'std_deviation_y': std_y.item(), 'std_deviation_z': std_z.item(),
        'median_x': med_x.item(), 'median_y': med_y.item(), 'median_z': med_z.item(),
        'between_x': between_x.item(), 'between_y': between_y.item(), 'between_z': between_z.item(),
        'sum_x': sum_x.item(), 'sum_y': sum_y.item(), 'sum_z': sum_z.item(),
        'sma_x': sma_x.item(), 'sma_y': sma_y.item(), 'sma_z': sma_z.item(),
        'count_x': int(count_x), 'count_y': int(count_y), 'count_z': int(count_z),
        'peak_to_peak_x': p2p_x.item(), 'peak_to_peak_y': p2p_y.item(), 'peak_to_peak_z': p2p_z.item(),
        'skewness_x': skew_x.item(), 'skewness_y': skew_y.item(), 'skewness_z': skew_z.item(),
        'kurtosis_x': kurt_x.item(), 'kurtosis_y': kurt_y.item(), 'kurtosis_z': kurt_z.item(),
        'rms_x': rms_x.item(), 'rms_y': rms_y.item(), 'rms_z': rms_z.item(),
        'zcr_x': float(zcr_x), 'zcr_y': float(zcr_y), 'zcr_z': float(zcr_z),
        'sma_global': sma_global.item(),
        'max_index_x': int(max_ix_x), 'max_index_y': int(max_ix_y), 'max_index_z': int(max_ix_z),
        'min_index_x': int(min_ix_x), 'min_index_y': int(min_ix_y), 'min_index_z': int(min_ix_z),
        'dominant_freq_x': int(dom_fx), 'dominant_freq_y': int(dom_fy), 'dominant_freq_z': int(dom_fz),
    }
    return pd.DataFrame([data_dic])


# -----------------------------
# Read one raw id file and return (seq_len x 3) float32 DataFrame
# Handles:
#   - Type1: columns like ["x [m]","y [m]","z [m]"] (3 columns)
#   - Type2: ["measurement type","x","y","z"] -> keep only acceleration rows
# -----------------------------
def load_xyz(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path).dropna()
    if df.shape[1] == 3:
        # Already x,y,z (sometimes named x [m] etc.)
        return df
    # Type2 with measurement type column
    if "measurement type" in df.columns:
        df = df[df["measurement type"] == "acceleration [m/s/s]"][["x", "y", "z"]]
        return df
    # Fallback: keep last 3 numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 3:
        return df[num_cols[-3:]]
    raise ValueError(f"Cannot parse raw file {file_path} with columns {df.columns}")


# -----------------------------
# Per-id feature extraction
# Returns numpy vector: [cnn_logits(18) + engineered_features(K)]
# -----------------------------
@torch.no_grad()
def extract_one_feature(
    file_path: Path,
    cnn: MultivariateCNN,
    target_len: int,
    device: torch.device,
    minmax_tensors_tuple,
) -> np.ndarray:
    min1, max1, min2, max2 = minmax_tensors_tuple

    xyz = load_xyz(file_path)
    # Identify type via original shape/columns (type2 has measurement type previously)
    # We'll infer by looking at original file columns count
    original = pd.read_csv(file_path, nrows=5)
    is_type2 = ("measurement type" in original.columns) or (original.shape[1] != 3)

    # For engineered stats: use raw acceleration axes as torch tensors
    if is_type2:
        # Load again but already acceleration filtered in load_xyz
        x_t = torch.tensor(xyz["x"].values, dtype=torch.float32)
        y_t = torch.tensor(xyz["y"].values, dtype=torch.float32)
        z_t = torch.tensor(xyz["z"].values, dtype=torch.float32)
    else:
        # Could be x [m],y [m],z [m] or x,y,z; take first 3 cols
        cols = xyz.columns.tolist()
        x_t = torch.tensor(xyz[cols[0]].values, dtype=torch.float32)
        y_t = torch.tensor(xyz[cols[1]].values, dtype=torch.float32)
        z_t = torch.tensor(xyz[cols[2]].values, dtype=torch.float32)

    feat_df = convert_to_features(x_t, y_t, z_t)
    engineered = feat_df.values.astype(np.float32).ravel()

    # Pad/cut for CNN input
    if len(xyz) < target_len:
        xyz = pad_sequence_df(xyz, target_len)
    else:
        xyz = xyz.iloc[:target_len].copy()

    x = torch.tensor(xyz.values, dtype=torch.float32, device=device)  # (L, 3)

    # Normalize
    if is_type2:
        x = (x - min2) / (max2 - min2 + 1e-6)
    else:
        x = (x - min1) / (max1 - min1 + 1e-6)

    x = x.transpose(0, 1).unsqueeze(0)  # (1, 3, L)
    logits = cnn(x).squeeze(0).detach().cpu().numpy().astype(np.float32)  # (18,)
    return np.concatenate([logits, engineered], axis=0)


# -----------------------------
# Build pseudo labels by filling gaps ONLY when same label on both sides
# -----------------------------
def build_pseudo_labels_strict_same(train_df: pd.DataFrame, cutoff_id: int) -> dict:
    # Map known labels
    known = train_df[["id", "activity"]].drop_duplicates("id")
    known = known.sort_values("id")
    label_by_id = dict(zip(known["id"].astype(int).tolist(), known["activity"].tolist()))

    # Fill gaps where (id_i, id_{i+1}) have same label
    ids = known["id"].astype(int).tolist()
    acts = known["activity"].tolist()

    pseudo = dict(label_by_id)
    for i in range(len(ids) - 1):
        a_id, b_id = ids[i], ids[i + 1]
        if a_id >= cutoff_id:
            break
        if b_id >= cutoff_id:
            b_id = cutoff_id
        if acts[i] == acts[i + 1] and (b_id - a_id) > 1:
            for missing in range(a_id + 1, b_id):
                pseudo[missing] = acts[i]
    # Keep only < cutoff_id
    pseudo = {k: v for k, v in pseudo.items() if k < cutoff_id}
    return pseudo


# -----------------------------
# Sequence dataset: predict next label from last seq_len features
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx: idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(int(y), dtype=torch.long)


# -----------------------------
# LSTM model
# -----------------------------
class SequenceLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_classes: int = 18, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data folder that contains train.csv, sample_submission.csv, and unlabeled/unlabeled/*.csv")
    parser.add_argument("--raw_dir", type=str, default="unlabeled/unlabeled",
                        help="Relative path from data_dir to raw files dir (default: unlabeled/unlabeled)")
    parser.add_argument("--cutoff_id", type=int, default=77000,
                        help="Train sequence cutoff id (use 77000 like your report)")
    parser.add_argument("--target_len", type=int, default=4000,
                        help="Pad/cut all raw sequences to this length (default 4000)")
    parser.add_argument("--seq_len", type=int, default=20,
                        help="How many past steps the LSTM sees (default 20)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_features", action="store_true",
                        help="Cache extracted features to features_cache/ for faster reruns")
    parser.add_argument("--cache_name", type=str, default="features",
                        help="Cache prefix name")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("✅ device:", device)

    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    sub_csv = data_dir / "sample_submission.csv"
    raw_dir = data_dir / args.raw_dir

    assert train_csv.exists(), f"train.csv not found at {train_csv}"
    assert sub_csv.exists(), f"sample_submission.csv not found at {sub_csv}"
    assert raw_dir.exists(), f"raw dir not found at {raw_dir}"

    train_df = pd.read_csv(train_csv)
    sub_df = pd.read_csv(sub_csv)

    classes, act2id, id2act = activity_mapping_from_submission(sub_df)
    num_classes = len(classes)
    print("✅ num_classes:", num_classes)

    # Build pseudo labels for ids < cutoff_id
    pseudo_label_str = build_pseudo_labels_strict_same(train_df, cutoff_id=args.cutoff_id)
    print(f"✅ pseudo labels available for {len(pseudo_label_str):,} ids < {args.cutoff_id}")

    # We will extract features for ALL ids needed:
    # - ids 0..cutoff_id-1 (context + training)
    # - test ids from submission (for prediction)
    seq_ids = list(range(0, args.cutoff_id))
    test_ids = sub_df["sample_id"].astype(int).tolist()
    needed_ids = sorted(set(seq_ids + test_ids))
    print(f"✅ total ids to feature-extract: {len(needed_ids):,}")

    # Caching
    cache_dir = data_dir / "features_cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_path = cache_dir / f"{args.cache_name}_{args.cutoff_id}_L{args.target_len}.parquet"

    if args.cache_features and cache_path.exists():
        feats_df = pd.read_parquet(cache_path)
        print("✅ loaded cached features:", cache_path)
    else:
        # CNN model (random init by default; you can load a trained CNN checkpoint if you have one)
        cnn = MultivariateCNN(num_channels=3, input_length=args.target_len, num_classes=num_classes).to(device)
        cnn.eval()

        mm = min_max_tensors(device)
        rows = []
        t0 = time.time()
        for i, sid in enumerate(needed_ids):
            fp = raw_dir / f"{sid}.csv"
            if not fp.exists():
                # If some ids do not exist on disk, skip
                continue
            vec = extract_one_feature(fp, cnn, args.target_len, device, mm)
            rows.append((sid, vec))
            if (i + 1) % 500 == 0:
                dt = time.time() - t0
                print(f"  extracted {i+1}/{len(needed_ids)}  ({dt:.1f}s)")
        if not rows:
            raise RuntimeError("No features extracted. Check --data_dir and --raw_dir.")
        feat_dim = rows[0][1].shape[0]
        feats_df = pd.DataFrame({
            "id": [r[0] for r in rows],
        })
        feat_mat = np.stack([r[1] for r in rows], axis=0)
        for j in range(feat_dim):
            feats_df[f"f{j}"] = feat_mat[:, j]
        if args.cache_features:
            feats_df.to_parquet(cache_path, index=False)
            print("✅ cached features to:", cache_path)

    # Build ordered feature matrix for sequence ids 0..cutoff_id-1
    feats_df = feats_df.sort_values("id").reset_index(drop=True)
    feats_df = feats_df.set_index("id")

    missing_seq = [sid for sid in seq_ids if sid not in feats_df.index]
    if missing_seq:
        print(f"⚠️ Missing {len(missing_seq)} raw files in sequence range 0..{args.cutoff_id-1}. "
              f"Example missing ids: {missing_seq[:10]}")

    # Only keep ids that exist on disk
    seq_ids_present = [sid for sid in seq_ids if sid in feats_df.index]
    X_seq = feats_df.loc[seq_ids_present].filter(regex=r"^f\d+$").values.astype(np.float32)

    # Labels for those ids (only if pseudo label exists)
    y_seq = np.full((len(seq_ids_present),), fill_value=-1, dtype=np.int64)
    for i, sid in enumerate(seq_ids_present):
        if sid in pseudo_label_str:
            y_seq[i] = act2id[pseudo_label_str[sid]]

    labeled_mask = y_seq != -1
    labeled_count = int(labeled_mask.sum())
    print(f"✅ labeled steps in sequence (after strict fill): {labeled_count:,} / {len(y_seq):,}")

    # We must train on contiguous regions where labels exist.
    # We'll take the longest contiguous labeled prefix as training.
    # (This mirrors your "sequence until row 77000" idea.)
    first_unlabeled = np.argmax(~labeled_mask) if (~labeled_mask).any() else len(labeled_mask)
    X_train_full = X_seq[:first_unlabeled]
    y_train_full = y_seq[:first_unlabeled]
    print(f"✅ using contiguous labeled prefix length: {len(y_train_full):,}")

    if len(y_train_full) < args.seq_len + 100:
        raise RuntimeError(
            f"Not enough labeled contiguous data to train sequence model. "
            f"Try a smaller cutoff_id, or a different fill strategy, or ensure raw files exist."
        )

    # Time-based split
    val_len = min(5000, len(y_train_full)//10)
    X_train, y_train = X_train_full[:-val_len], y_train_full[:-val_len]
    X_val, y_val = X_train_full[-val_len:], y_train_full[-val_len:]

    train_ds = SeqDataset(X_train, y_train, seq_len=args.seq_len)
    val_ds   = SeqDataset(X_val, y_val,   seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Train LSTM
    model = SequenceLSTM(input_dim=X_seq.shape[1], hidden_dim=256, num_layers=2, num_classes=num_classes, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_path = data_dir / "seq_lstm.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)
            tr_n += xb.size(0)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
                va_n += xb.size(0)

        tr_loss /= max(tr_n, 1)
        va_loss /= max(va_n, 1)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"✅ saved best model -> {best_path} (val_loss={best_val:.4f})")

    # Load best model for inference
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Predict for submission ids
    # We'll do sequential inference using the feature sequence:
    # for each test id, we take its index in the sorted ids of ALL needed ids,
    # and build a window of last seq_len steps.
    all_ids_present = sorted(feats_df.index.astype(int).tolist())
    id_to_pos = {sid: i for i, sid in enumerate(all_ids_present)}
    X_all = feats_df.loc[all_ids_present].filter(regex=r"^f\d+$").values.astype(np.float32)

    def predict_probs_for_id(target_id: int) -> np.ndarray:
        if target_id not in id_to_pos:
            return np.ones((num_classes,), dtype=np.float32) / num_classes
        pos = id_to_pos[target_id]
        start = max(0, pos - args.seq_len)
        window = X_all[start:pos]
        # pad at the front if needed
        if window.shape[0] < args.seq_len:
            pad = np.zeros((args.seq_len - window.shape[0], window.shape[1]), dtype=np.float32)
            window = np.vstack([pad, window])
        window = torch.tensor(window[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(window).squeeze(0)
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float32)
        return probs

    out_rows = []
    for sid in test_ids:
        probs = predict_probs_for_id(int(sid))
        row = {"sample_id": int(sid)}
        for i, cls in enumerate(classes):
            row[cls] = float(probs[i])
        out_rows.append(row)

    submission = pd.DataFrame(out_rows, columns=["sample_id"] + classes)
    out_path = data_dir / "submission_seq_lstm.csv"
    submission.to_csv(out_path, index=False)
    print("✅ wrote submission ->", out_path)


if __name__ == "__main__":
    main()
