# ASL Recognition with MediaPipe and Random Forest

This project provides a complete pipeline for recognizing American Sign Language (ASL) alphabet gestures in real-time. It includes data preprocessing, landmark extraction, feature normalization, model training, evaluation, and a live demo with stable-letter detection.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Project Structure](#project-structure)
5. [Setup Instructions](#setup-instructions)
6. [Dataset Preparation](#dataset-preparation)
7. [Landmark Extraction](#landmark-extraction)
8. [Data Normalization](#data-normalization)
9. [Model Training](#model-training)
10. [Model Evaluation](#model-evaluation)
11. [Live Demo](#live-demo)
12. [Troubleshooting](#troubleshooting)
13. [License](#license)

---

## Overview

This project demonstrates how to build a hand-gesture recognition system for ASL alphabet using MediaPipe for hand landmark detection and a Random Forest classifier for classification. The system can:

* Extract 21 hand landmarks from dataset images.
* Normalize landmarks to a wrist-origin, fingertip-scaled coordinate system.
* Train a Random Forest model saved as `asl_rf_model.pkl`.
* Evaluate performance on a held-out test set.
* Run a live webcam demo that captures and appends letters when held steady for 2 seconds, with an audible beep.

## Features

* **Dataset Processing**: Converts raw ASL images into 63-dimensional landmark feature vectors.
* **Normalization**: Centers hand landmarks at the wrist and scales by the middle fingertip distance.
* **Model Training**: Trains a Random Forest with hyperparameter tuning to reduce overfitting.
* **Live Inference**: Real-time prediction with stable-letter detection and audible feedback.

## Prerequisites

* Python 3.8+
* pip package manager
* Webcam for live demo

## Project Structure

```
├── asl_landmark_features.csv       # Raw CSV of extracted landmarks
├── train_model.ipynb               # Jupyter notebook with training & evaluation steps, Landmark extraction script, Real-time inference script
├── asl_rf_model.pkl                # Saved Random Forest model (After running the code the file will be automatically generated)
└── README.md                       # This documentation
```

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies**

   ```bash
   pip install opencv-python mediapipe scikit-learn pandas numpy joblib tqdm
   ```

## Dataset Preparation

1. Download the “ASL Alphabet” Kaggle dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data?select=asl_alphabet_train
2. Unzip into `data/asl_alphabet_train/<ClassName>/...` and `data/asl_alphabet_test/`.
3. Run `extract_landmarks.py` or the notebook cell to generate `asl_landmark_features.csv`.

## Landmark Extraction

* Uses MediaPipe Hands in static mode to detect 21 landmarks per image.
* Outputs 63 features `[lm0_x, lm0_y, lm0_z, ..., lm20_x, lm20_y, lm20_z]` plus label.

Example code snippet:

```python
from extract_landmarks import extract_landmarks
X, y = extract_landmarks('data/asl_alphabet_train')
```

## Data Normalization

* Centers coordinates at wrist landmark (`lm0`).
* Scales by distance between wrist and middle fingertip (`lm12`).
* Saves result to `asl_landmark_normalized.csv`.

## Model Training

1. Load `asl_landmark_normalized.csv` in `train_model.ipynb`.
2. Encode labels with `LabelEncoder()`.
3. Split into train/test (stratified, 80/20).
4. Train `RandomForestClassifier(n_estimators=150, max_depth=20, min_samples_split=2)`.
5. Save model:

   ```python
   import joblib
   joblib.dump(classifier, 'asl_rf_model.pkl')
   ```

## Model Evaluation

* Evaluate on held-out test set using `accuracy_score`, `confusion_matrix`, and `classification_report`.
* Ensure encoding consistency (test labels encoded with same `LabelEncoder`).

## Live Demo

Run `live_demo.py`:

```bash
python live_demo.py
```

* Hold an ASL sign steadily for 2 seconds to capture it.
* Beep confirms capture; letters append to on-screen buffer.
* Press `Esc` to exit.

## Troubleshooting

* **0% Test Accuracy**: Ensure test labels are encoded with the same encoder used in training and drop unmatched classes.
* **Overfitting**: Use grid search or reduce tree depth, increase `min_samples_leaf`, or try ensemble stacking.
* **MediaPipe Errors**: Verify correct installation and that images contain visible hands.

## License

This project is released under the MIT License.
