import os
import time

import pandas as pd

from models_utils.GLOBALS import *
from models_utils.Datasets import pad_sequence


def convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor):
    """
    Calculates differents features for a x,y,z data
    :param data_x_tensor: x data
    :param data_y_tensor: y data
    :param data_z_tensor: z data
    :return: dictionary with features extracted from data
    """

    def manual_skewness(data):
        n = len(data)
        mean = torch.mean(data)
        std_dev = torch.std(data, unbiased=True)
        skewness = (n / ((n - 1) * (n - 2))) * torch.sum(((data - mean) / std_dev) ** 3)
        return skewness

    def manual_kurtosis(data):
        n = len(data)
        mean = torch.mean(data)
        std_dev = torch.std(data, unbiased=True)
        kurtosis = (n * (n + 1) * torch.sum(((data - mean) / std_dev) ** 4) / ((n - 1) * (n - 2) * (n - 3))) - (
                3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return kurtosis

    mean_x, mean_y, mean_z = torch.mean(data_x_tensor), torch.mean(data_y_tensor), torch.mean(data_z_tensor)
    std_deviation_x, std_deviation_y, std_deviation_z = torch.std(data_x_tensor), torch.std(data_y_tensor), torch.std(
        data_z_tensor)
    median_x, median_y, median_z = torch.median(data_x_tensor), torch.median(data_y_tensor), torch.median(data_z_tensor)
    between_x, between_y, between_z = (torch.max(data_x_tensor) - torch.min(data_x_tensor)) / len(data_x_tensor), (
            torch.max(data_y_tensor) - torch.min(data_y_tensor)) / len(data_y_tensor), (
                                              torch.max(data_z_tensor) - torch.min(data_z_tensor)) / len(
        data_z_tensor)
    sum_x, sum_y, sum_z = torch.sum(data_x_tensor), torch.sum(data_y_tensor), torch.sum(data_z_tensor)
    sma_x, sma_y, sma_z = torch.mean(torch.abs(data_x_tensor - torch.mean(data_x_tensor))), torch.mean(
        torch.abs(data_y_tensor - torch.mean(data_y_tensor))), torch.mean(
        torch.abs(data_z_tensor - torch.mean(data_z_tensor)))
    count_x, count_y, count_z = len(data_x_tensor), len(data_y_tensor), len(data_z_tensor)
    peak_to_peak_x, peak_to_peak_y, peak_to_peak_z = torch.max(data_x_tensor) - torch.min(data_x_tensor), torch.max(
        data_y_tensor) - torch.min(data_y_tensor), torch.max(data_z_tensor) - torch.min(data_z_tensor)

    skewness_x, skewness_y, skewness_z = manual_skewness(data_x_tensor), manual_skewness(
        data_y_tensor), manual_skewness(data_z_tensor)
    kurtosis_x, kurtosis_y, kurtosis_z = manual_kurtosis(data_x_tensor), manual_kurtosis(
        data_y_tensor), manual_kurtosis(data_z_tensor)
    rms_x, rms_y, rms_z = torch.sqrt(torch.mean(data_x_tensor ** 2)), torch.sqrt(
        torch.mean(data_y_tensor ** 2)), torch.sqrt(torch.mean(data_z_tensor ** 2))
    zcr_x, zcr_y, zcr_z = ((data_x_tensor[:-1] * data_x_tensor[1:]) < 0).sum(), (
            (data_y_tensor[:-1] * data_y_tensor[1:]) < 0).sum(), (
            (data_z_tensor[:-1] * data_z_tensor[1:]) < 0).sum()
    sma = torch.mean(torch.abs(data_x_tensor) + torch.abs(data_y_tensor) + torch.abs(data_z_tensor))
    max_index_x, max_index_y, max_index_z = torch.argmax(data_x_tensor), torch.argmax(data_y_tensor), torch.argmax(
        data_z_tensor)
    min_index_x, min_index_y, min_index_z = torch.argmin(data_x_tensor), torch.argmin(data_y_tensor), torch.argmin(
        data_z_tensor)
    fft_x, fft_y, fft_z = torch.fft.fft(data_x_tensor), torch.fft.fft(data_y_tensor), torch.fft.fft(data_z_tensor)
    dominant_freq_x, dominant_freq_y, dominant_freq_z = torch.argmax(torch.abs(fft_x)), torch.argmax(
        torch.abs(fft_y)), torch.argmax(torch.abs(fft_z))

    data_dic = {
        'mean_x': mean_x.item(), 'mean_y': mean_y.item(), 'mean_z': mean_z.item(),
        'std_deviation_x': std_deviation_x.item(), 'std_deviation_y': std_deviation_y.item(),
        'std_deviation_z': std_deviation_z.item(),
        'median_x': median_x.item(), 'median_y': median_y.item(), 'median_z': median_z.item(),
        'between_x': between_x.item(), 'between_y': between_y.item(), 'between_z': between_z.item(),
        'sum_x': sum_x.item(), 'sum_y': sum_y.item(), 'sum_z': sum_z.item(),
        'sma_x': sma_x.item(), 'sma_y': sma_y.item(), 'sma_z': sma_z.item(),
        'count_x': count_x, 'count_y': count_y, 'count_z': count_z,
        'peak_to_peak_x': peak_to_peak_x.item(), 'peak_to_peak_y': peak_to_peak_y.item(),
        'peak_to_peak_z': peak_to_peak_z.item(),
        'skewness_x': skewness_x.item(), 'skewness_y': skewness_y.item(), 'skewness_z': skewness_z.item(),
        'kurtosis_x': kurtosis_x.item(), 'kurtosis_y': kurtosis_y.item(), 'kurtosis_z': kurtosis_z.item(),
        'rms_x': rms_x.item(), 'rms_y': rms_y.item(), 'rms_z': rms_z.item(),
        'zcr_x': zcr_x.item(), 'zcr_y': zcr_y.item(), 'zcr_z': zcr_z.item(),
        'sma_global': sma.item(),
        'max_index_x': max_index_x.item(), 'max_index_y': max_index_y.item(), 'max_index_z': max_index_z.item(),
        'min_index_x': min_index_x.item(), 'min_index_y': min_index_y.item(), 'min_index_z': min_index_z.item(),
        'dominant_freq_x': dominant_freq_x.item(), 'dominant_freq_y': dominant_freq_y.item(),
        'dominant_freq_z': dominant_freq_z.item()
    }

    return pd.DataFrame([data_dic])


def get_results(model_type1, model_type2, classifier_type1, classifier_type2, label_encoder_type1, label_encoder_type2,
                target_size_type1, target_size_type2, cols_to_drop_type_1, cols_to_drop_type_2, embedding_names):
    """
    Get final results on test data
    :param model_type1: model for first type of data
    :param model_type2: model for second type of data
    :param classifier_type1: classifier for first type of data
    :param classifier_type2: classifier for second type of data
    :param label_encoder_type1: label encoder for first type of data
    :param label_encoder_type2: label encoder for second type of data
    :param target_size_type1: max size of first type of data
    :param target_size_type2: max size of second type of data
    :param cols_to_drop_type_1: cols to drop from first type of data
    :param cols_to_drop_type_2: cols to drop from second type of data
    :param embedding_names: name for embedding columns
    :return: results dataframe
    """
    results_list = []
    sample_submission_path = os.path.join(DATA_DIR, "sample_submission.csv")
    test_data = pd.read_csv(sample_submission_path)['sample_id'].to_list()
    len_test_data = len(test_data)
    start_time = time.time()
    for i, file_id in enumerate(test_data):
        class_path = os.path.join(files_directory, f"{file_id}.csv")
        new_data = pd.read_csv(class_path).dropna()
        with torch.no_grad():
            if new_data.shape[1] == 3:
                data_x_tensor = torch.tensor(new_data["x [m]"].values, dtype=torch.float32)
                data_y_tensor = torch.tensor(new_data["y [m]"].values, dtype=torch.float32)
                data_z_tensor = torch.tensor(new_data["z [m]"].values, dtype=torch.float32)
                new_features = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)
                if len(new_data) < target_size_type1:
                    new_data = pad_sequence(new_data, target_size_type1)
                elif len(new_data) > target_size_type1:
                    new_data = new_data[:target_size_type1]
                new_data = torch.tensor(new_data.values, dtype=torch.float32).to(device)
                normalized_new_data = (new_data - min_values_type1) / (max_values_type1 - min_values_type1 + 1e-6)
                # shape: (seq_len, 3)
                if normalized_new_data.dim() == 2:
                    normalized_new_data = normalized_new_data.unsqueeze(0)  # (1, seq_len, 3)

                logits = model_type1.encode(normalized_new_data).squeeze(0).detach().cpu().numpy()

                res = pd.DataFrame([logits], columns=embedding_names)
                for col, value in new_features.items():
                    res[col] = value
                res = res.drop(columns=cols_to_drop_type_1[1:])
                res_prob = classifier_type1.predict_proba(res)[0]
                res_dict = dict(zip(label_encoder_type1.inverse_transform(classifier_type1.classes_), res_prob))

            else:
                new_data = new_data[new_data.iloc[:, 0] == 'acceleration [m/s/s]'].iloc[:, 1:]
                data_x_tensor = torch.tensor(new_data["x"].values, dtype=torch.float32)
                data_y_tensor = torch.tensor(new_data["y"].values, dtype=torch.float32)
                data_z_tensor = torch.tensor(new_data["z"].values, dtype=torch.float32)
                new_features = convert_to_features(data_x_tensor, data_y_tensor, data_z_tensor)
                if len(new_data) < target_size_type2:
                    new_data = pad_sequence(new_data, target_size_type2)
                elif len(new_data) > target_size_type2:
                    new_data = new_data[:target_size_type2]
                new_data = torch.tensor(new_data.values, dtype=torch.float32).to(device)
                normalized_new_data = (new_data - min_values_type2) / (max_values_type2 - min_values_type2 + 1e-6)
                # shape: (seq_len, 3)
                if normalized_new_data.dim() == 2:
                    normalized_new_data = normalized_new_data.unsqueeze(0)  # (1, seq_len, 3)

                logits = model_type2.encode(normalized_new_data).squeeze(0).detach().cpu().numpy()

                res = pd.DataFrame([logits], columns=embedding_names)
                for col, value in new_features.items():
                    res[col] = value
                res = res.drop(columns=cols_to_drop_type_2[1:])
                res_prob = classifier_type2.predict_proba(res)[0]
                res_dict = dict(zip(label_encoder_type2.inverse_transform(classifier_type2.classes_), res_prob))

        result_dict = {label: res_dict.get(label, 0) for label in activity_id_mapping.keys()}
        result_dict['sample_id'] = file_id
        results_list.append(result_dict)

        if (i + 1) % 100 == 0:
            batch_end_time = time.time()
            print(f'Processed {i + 1}/{len_test_data} files in {batch_end_time - start_time:.2f} seconds.')
            start_time = time.time()

    results = pd.DataFrame(results_list, columns=['sample_id'] + list(activity_id_mapping.keys()))
    return results
