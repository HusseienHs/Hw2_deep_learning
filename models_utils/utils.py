import os
import time
import torch
import pandas as pd

from models_utils.GLOBALS import *
from models_utils.Datasets import pad_sequence


def convert_to_features(data_x, data_y, data_z):
    """
    Extract statistical features from 3-axis sensor data.
    """

    def skew(x):
        n = len(x)
        m = torch.mean(x)
        s = torch.std(x, unbiased=True)
        return (n / ((n - 1) * (n - 2))) * torch.sum(((x - m) / s) ** 3)

    def kurt(x):
        n = len(x)
        m = torch.mean(x)
        s = torch.std(x, unbiased=True)
        return (n * (n + 1) * torch.sum(((x - m) / s) ** 4) / ((n - 1) * (n - 2) * (n - 3))) - \
               (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))

    mean_x, mean_y, mean_z = torch.mean(data_x), torch.mean(data_y), torch.mean(data_z)
    std_x, std_y, std_z = torch.std(data_x), torch.std(data_y), torch.std(data_z)
    med_x, med_y, med_z = torch.median(data_x), torch.median(data_y), torch.median(data_z)

    rng_x = (torch.max(data_x) - torch.min(data_x)) / len(data_x)
    rng_y = (torch.max(data_y) - torch.min(data_y)) / len(data_y)
    rng_z = (torch.max(data_z) - torch.min(data_z)) / len(data_z)

    sum_x, sum_y, sum_z = torch.sum(data_x), torch.sum(data_y), torch.sum(data_z)
    sma_x = torch.mean(torch.abs(data_x - torch.mean(data_x)))
    sma_y = torch.mean(torch.abs(data_y - torch.mean(data_y)))
    sma_z = torch.mean(torch.abs(data_z - torch.mean(data_z)))

    cnt_x, cnt_y, cnt_z = len(data_x), len(data_y), len(data_z)

    ptp_x = torch.max(data_x) - torch.min(data_x)
    ptp_y = torch.max(data_y) - torch.min(data_y)
    ptp_z = torch.max(data_z) - torch.min(data_z)

    skew_x, skew_y, skew_z = skew(data_x), skew(data_y), skew(data_z)
    kurt_x, kurt_y, kurt_z = kurt(data_x), kurt(data_y), kurt(data_z)

    rms_x = torch.sqrt(torch.mean(data_x ** 2))
    rms_y = torch.sqrt(torch.mean(data_y ** 2))
    rms_z = torch.sqrt(torch.mean(data_z ** 2))

    zcr_x = ((data_x[:-1] * data_x[1:]) < 0).sum()
    zcr_y = ((data_y[:-1] * data_y[1:]) < 0).sum()
    zcr_z = ((data_z[:-1] * data_z[1:]) < 0).sum()

    sma_global = torch.mean(torch.abs(data_x) + torch.abs(data_y) + torch.abs(data_z))

    max_x, max_y, max_z = torch.argmax(data_x), torch.argmax(data_y), torch.argmax(data_z)
    min_x, min_y, min_z = torch.argmin(data_x), torch.argmin(data_y), torch.argmin(data_z)

    fft_x, fft_y, fft_z = torch.fft.fft(data_x), torch.fft.fft(data_y), torch.fft.fft(data_z)
    dom_x, dom_y, dom_z = torch.argmax(torch.abs(fft_x)), torch.argmax(torch.abs(fft_y)), torch.argmax(torch.abs(fft_z))

    return pd.DataFrame([{
        "mean_x": mean_x.item(), "mean_y": mean_y.item(), "mean_z": mean_z.item(),
        "std_deviation_x": std_x.item(), "std_deviation_y": std_y.item(), "std_deviation_z": std_z.item(),
        "median_x": med_x.item(), "median_y": med_y.item(), "median_z": med_z.item(),
        "between_x": rng_x.item(), "between_y": rng_y.item(), "between_z": rng_z.item(),
        "sum_x": sum_x.item(), "sum_y": sum_y.item(), "sum_z": sum_z.item(),
        "sma_x": sma_x.item(), "sma_y": sma_y.item(), "sma_z": sma_z.item(),
        "count_x": cnt_x, "count_y": cnt_y, "count_z": cnt_z,
        "peak_to_peak_x": ptp_x.item(), "peak_to_peak_y": ptp_y.item(), "peak_to_peak_z": ptp_z.item(),
        "skewness_x": skew_x.item(), "skewness_y": skew_y.item(), "skewness_z": skew_z.item(),
        "kurtosis_x": kurt_x.item(), "kurtosis_y": kurt_y.item(), "kurtosis_z": kurt_z.item(),
        "rms_x": rms_x.item(), "rms_y": rms_y.item(), "rms_z": rms_z.item(),
        "zcr_x": zcr_x.item(), "zcr_y": zcr_y.item(), "zcr_z": zcr_z.item(),
        "sma_global": sma_global.item(),
        "max_index_x": max_x.item(), "max_index_y": max_y.item(), "max_index_z": max_z.item(),
        "min_index_x": min_x.item(), "min_index_y": min_y.item(), "min_index_z": min_z.item(),
        "dominant_freq_x": dom_x.item(), "dominant_freq_y": dom_y.item(), "dominant_freq_z": dom_z.item(),
    }])


def get_results(
    model_type1,
    model_type2,
    classifier_type1,
    classifier_type2,
    label_encoder_type1,
    label_encoder_type2,
    target_size_type1,
    target_size_type2,
    cols_to_drop_type_1,
    cols_to_drop_type_2,
    embedding_names,
):
    results = []
    sample_ids = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))["sample_id"].tolist()

    start_time = time.time()

    for idx, sample_id in enumerate(sample_ids):
        file_path = os.path.join(files_directory, f"{sample_id}.csv")
        df = pd.read_csv(file_path).dropna()

        with torch.no_grad():
            if df.shape[1] == 3:
                x = torch.tensor(df["x [m]"].values, dtype=torch.float32)
                y = torch.tensor(df["y [m]"].values, dtype=torch.float32)
                z = torch.tensor(df["z [m]"].values, dtype=torch.float32)

                feats = convert_to_features(x, y, z)

                if len(df) < target_size_type1:
                    df = pad_sequence(df, target_size_type1)
                else:
                    df = df.iloc[:target_size_type1]

                tensor = torch.tensor(df.values, dtype=torch.float32)
                tensor = (tensor - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
                tensor = tensor.unsqueeze(0)

                emb = model_type1.encode(tensor).squeeze(0).cpu().numpy()
                row = pd.DataFrame([emb], columns=embedding_names)

                for k, v in feats.items():
                    row[k] = v

                row = row.drop(columns=cols_to_drop_type_1[1:])
                probs = classifier_type1.predict_proba(row)[0]
                pred_dict = dict(zip(label_encoder_type1.inverse_transform(classifier_type1.classes_), probs))

            else:
                df = df[df.iloc[:, 0] == "acceleration [m/s/s]"].iloc[:, 1:]
                x = torch.tensor(df["x"].values, dtype=torch.float32)
                y = torch.tensor(df["y"].values, dtype=torch.float32)
                z = torch.tensor(df["z"].values, dtype=torch.float32)

                feats = convert_to_features(x, y, z)

                if len(df) < target_size_type2:
                    df = pad_sequence(df, target_size_type2)
                else:
                    df = df.iloc[:target_size_type2]

                tensor = torch.tensor(df.values, dtype=torch.float32)
                tensor = (tensor - min_values_type2) / (max_values_type2 - min_values_type2 + 1e-6)
                tensor = tensor.unsqueeze(0)

                emb = model_type2.encode(tensor).squeeze(0).cpu().numpy()
                row = pd.DataFrame([emb], columns=embedding_names)

                for k, v in feats.items():
                    row[k] = v

                row = row.drop(columns=cols_to_drop_type_2[1:])
                probs = classifier_type2.predict_proba(row)[0]
                pred_dict = dict(zip(label_encoder_type2.inverse_transform(classifier_type2.classes_), probs))

        result = {label: pred_dict.get(label, 0.0) for label in activity_id_mapping.keys()}
        result["sample_id"] = sample_id
        results.append(result)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(sample_ids)} samples in {time.time() - start_time:.2f}s")
            start_time = time.time()

    return pd.DataFrame(results)
