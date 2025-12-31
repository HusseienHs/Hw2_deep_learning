🧠 Human Activity Recognition (HAR) – Deep Learning & Machine Learning Project
📌 Overview

This repository contains a complete Human Activity Recognition (HAR) pipeline using sensor data collected from wearable devices.
The goal of the project is to accurately classify human activities such as walking, typing, brushing teeth, and more, using a combination of deep learning, classical machine learning, and feature engineering techniques.

The project explores multiple modeling strategies including CNNs, LSTMs, hybrid CNN–LSTM architectures, Random Forests, and ensemble-based approaches.

📂 Project Structure
Source Code/
├── CNN/
│   ├── CNN.py
│   └── cnn_utils.py
│
├── LSTM/
│   ├── lstm_autoencoder.py
│   └── lstm_autoencoders_utils.py
│
├── models_utils/
│   ├── Datasets.py
│   ├── GLOBALS.py
│   ├── utils.py
│
├── RF_XGB/
│   ├── RandomForest.py
│   └── XGBoost.py
│
├── main_models/
│   ├── only_cnn.ipynb
│   ├── only_1dcnn.ipynb
│   ├── lstm+cnn_rf.ipynb
│   ├── embedding_rf.ipynb
│   ├── lstm_secret_data.ipynb
│   ├── only_rf.ipynb
│   ├── only_xgboost.ipynb
│   └── simple_prob.ipynb
│
├── data/
│   ├── train.csv
│   ├── unlabeled/
│   └── sample_submission.csv
│
├── results/
│   ├── *.csv
│   ├── *.pth
│
├── setup_paths.py
└── README.md

🧠 Project Components
1️⃣ CNN Models

Located in CNN/

Implements 1D and 3D convolutional neural networks.

Used for direct feature extraction from raw time-series sensor data.

Supports both single-sensor and multi-sensor inputs.

2️⃣ LSTM Models

Located in LSTM/

Implements LSTM-based autoencoders.

Used for learning temporal representations from sensor sequences.

Can be used as standalone predictors or as feature extractors.

3️⃣ Feature Engineering

Located in models_utils/

Extracts statistical features (mean, std, skewness, kurtosis, etc.).

Handles normalization and data preprocessing.

Provides unified Dataset classes for PyTorch pipelines.

4️⃣ Classical Machine Learning

Located in RF_XGB/

Random Forest and XGBoost classifiers.

Used on top of extracted features or learned embeddings.

Often used as strong baselines or ensemble components.

5️⃣ Experiments & Pipelines

Located in main_models/

End-to-end experiment notebooks.

Includes CNN-only, LSTM-only, hybrid CNN+LSTM, and ensemble models.

Also includes scripts for generating final submission files.

📊 Dataset Description

The dataset contains human activity sensor recordings:

Type 1: Smartwatch accelerometer data (x, y, z).

Type 2: Multi-sensor data including acceleration signals.

Each file corresponds to a single activity instance.
The target label represents the performed activity (e.g., walking, typing, washing hands).

🧪 Model Training & Evaluation

Train/validation split is applied at the subject level.

Models are evaluated using classification accuracy and log loss.

Advanced techniques used:

Feature extraction with CNNs

Sequence modeling with LSTMs

Ensemble learning (RF + NN)

Calibration and probability refinement

🧠 Key Highlights

Hybrid CNN–LSTM architectures outperform standalone models.

Feature-based models (RF/XGBoost) are strong baselines.

Learned representations significantly improve performance.

Modular design enables easy experimentation and extension.

🚀 How to Run

Prepare data inside data/

Run feature extraction or model scripts inside main_models/

Train models and generate predictions

Export results as submission.csv

📌 Notes

All paths are handled via setup_paths.py

GPU acceleration supported (PyTorch)

Designed for reproducibility and scalability
