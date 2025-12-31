🧠 Human Activity Recognition (HAR) – Deep Learning & Machine Learning Project
📌 Overview

This repository contains a complete Human Activity Recognition (HAR) pipeline using sensor data collected from wearable devices.
The goal of the project is to accurately classify human activities such as walking, typing, brushing teeth, and more, using a combination of deep learning, classical machine learning, and feature engineering techniques.

The project explores multiple modeling strategies including CNNs, LSTMs, hybrid CNN–LSTM architectures, Random Forests, and ensemble-based approaches.

📂 Project Structure




📂 Project Structure

This repository is dedicated to the Human Activity Recognition (HAR) project, which aims to classify human activities using sensor data collected from wearable devices.
The project focuses on building and evaluating multiple deep learning and machine learning models to recognize activities such as walking, reading, using a phone, brushing teeth, and more.

📁 CNN

Contains convolutional neural network implementations used for feature extraction from raw time-series data.

CNN.py – CNN-based feature extractor

cnn_utils.py – Utility functions for CNN models

📁 LSTM

Implements LSTM-based sequence models and autoencoders.

lstm_autoencoder.py – LSTM autoencoder architecture

lstm_autoencoders_utils.py – Helper functions for LSTM training and inference

📁 main_models

Contains all main experiments and model pipelines.

cnn_to_rf.ipynb – 3D CNN feature extractor followed by Random Forest

cnn_to_xgb.ipynb – 3D CNN feature extractor with XGBoost

embedding_nn.ipynb – LSTM autoencoder embeddings with neural network classifier

embedding_rf.ipynb – LSTM embeddings with Random Forest

lstm+cnn_rf.ipynb – Combined CNN + LSTM feature extraction with Random Forest

lstm_secret_data.ipynb – 3D CNN trained on extended dataset with missing data recovery

only_1dcnn.ipynb – Pure 1D CNN model

only_cnn.ipynb – Pure 3D CNN model

only_rf.ipynb – Random Forest baseline

only_xgboost.ipynb – XGBoost-based classifier

simple_prob.ipynb – Probability-based baseline using class frequency

📁 main_utils

Utility scripts used across experiments.

fill_ranges_script.ipynb – Extends missing ranges in training data

generate_graphs.ipynb – Visualization and performance plots

get_all_secret_data.ipynb – Extracts features for hidden test data

get_secret_results.ipynb – Generates final predictions for submission

merge_lstm_results.ipynb – Ensemble method combining multiple LSTM outputs

📁 models_utils

Core utility modules used throughout the project.

Datasets.py – PyTorch dataset classes

GLOBALS.py – Global configuration and constants

utils.py – Feature extraction and helper utilities

📁 NN

Neural network utilities and helpers.

NeuralNetwork.py – General neural network architecture

nn_utils.py – Supporting utility functions

📁 RF_XGB

Classical machine learning models.

RandomForest.py – Random Forest classifier

XGBoost.py – XGBoost model implementation



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
