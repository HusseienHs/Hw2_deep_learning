🧠 Human Activity Recognition (HAR)

This project focuses on Human Activity Recognition (HAR) using wearable sensor data.
Multiple deep learning and classical machine learning models are implemented and compared to classify human activities such as walking, typing, brushing teeth, and more.

📌 Project Overview

The goal is to classify time-series sensor data collected from wearable devices using:

Deep learning models (CNN, LSTM, CNN–LSTM)

Classical ML models (Random Forest, XGBoost)

Pretrained time-series models (MOMENT)

The project explores feature extraction, temporal modeling, and representation learning for activity recognition.

📂 Project Structure



.
├── CNN/                    # CNN-based feature extractors



├── LSTM/                   # LSTM & autoencoder models



├── main_models/            # End-to-end training pipelines



├── models_utils/           # Dataset loaders & utilities



├── RF_XGB/                 # Random Forest & XGBoost models




├── NN/                     # Generic neural network utilities




├── data/                   # Input sensor data




└── README.md

🧠 Models Implemented
1️⃣ Deep Learning Models

1D CNN – captures local temporal patterns

LSTM / BiLSTM – models long-term temporal dependencies

CNN + LSTM hybrid – combines spatial + temporal features

MOMENT (Pretrained Transformer) – fine-tuned for time-series classification

2️⃣ Classical Machine Learning

Random Forest

XGBoost

Feature-based pipelines using statistical descriptors

📊 Dataset Description

Type: Multivariate time-series sensor data

Sensors: Accelerometer (x, y, z) and additional channels

Labels: Human activities (e.g., walking, typing, brushing teeth)

Splitting: Stratified train/validation split

Challenges:

Class imbalance

Overlapping motion patterns

Noisy and variable-length sequences

🧪 Training & Evaluation

Loss: Cross-entropy

Metrics: Accuracy, validation loss

Mixed precision (AMP) for faster training

Partial fine-tuning for large pretrained models

Gradient accumulation for memory efficiency

Evaluation includes:

Learning curves (train/val loss)

Class-wise behavior analysis

Comparison across model families

🔍 Key Findings

CNNs perform well on short, structured motion patterns.

LSTMs improve temporal understanding but require more data.

Pretrained models (MOMENT) provide strong representations but are computationally heavy.

Hybrid approaches (CNN + ML) offer an excellent accuracy–efficiency tradeoff.

🚀 How to Run

Prepare data in the data/ directory

Run desired experiment from main_models/

Adjust hyperparameters as needed

Evaluate results using validation metrics

📌 Notes

Designed for reproducibility and modular experimentation.

All paths and configurations are centralized.

Supports GPU acceleration.

🏁 Summary

This project demonstrates a full machine learning pipeline for human activity recognition, from data preprocessing to advanced deep learning models. It highlights trade-offs between accuracy, computational cost, and model complexity while providing a scalable and extensible experimental framework.
