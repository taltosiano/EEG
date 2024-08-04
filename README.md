# EEG Model Training and Evaluation

## Overview

This repository provides code for training and evaluating models for EEG data classification. It supports two types of models:
1. **EfficientNet** with attention mechanisms.
2. **SVM** (Support Vector Machine) classifier.

## Project Structure

- `dataloader.py`: Contains the `EEGDataset` class for loading and processing EEG data.
- `models_dir/models.py`: Defines model architectures, including `EffNetAttention` and `EEG_SVM_Classifier`.
- `traintest.py`: Implements functions for training (`train`) and validating (`validate`) models.
- `data_prep.py`: Contains functions for preparing and preprocessing data.
- `DataAnalysis.py`: Includes functions for analyzing the results and generating visualizations.
- `main.py`: The main script for argument parsing, model initialization, and execution of training and evaluation.

## Code Description

### `dataloader.py`

- **`EEGDataset`**: This class is responsible for loading and processing EEG data. It manages the data pipeline, including:
  - Reading data from JSON files.
  - Applying necessary transformations.
  - Preparing data batches for training and evaluation.

### `models_dir/models.py`

- **`EffNetAttention`**: A model based on EfficientNet with additional attention mechanisms. It features:
  - Optional use of pre-trained EfficientNet weights.
  - Integration of attention layers to enhance feature extraction and representation.
  
- **`EEG_SVM_Classifier`**: An SVM classifier tailored for EEG data. It supports:
  - Various kernel types (`linear`, `poly`, `rbf`, `sigmoid`).
  - Methods for training and evaluating the SVM model on EEG data.

### `traintest.py`

- **`train`**: This function handles the training of the EEG model, including:
  - The training loop with forward and backward passes.
  - Optimization and loss computation.
  - Saving model checkpoints and tracking training progress.
  
- **`validate`**: This function evaluates the model on validation or test datasets. It computes:
  - Metrics such as mAP (mean Average Precision) and AUC (Area Under the Curve).
  - Saves prediction results for further analysis.

### `data_prep.py`

- **Functions**: Contains various utility functions for preparing and preprocessing EEG data. This includes:
  - Data normalization.
  - Splitting data into training, validation, and testing sets.
  - Other necessary preprocessing steps before model training.

### `DataAnalysis.py`

- **Functions**: Provides functions for analyzing the results of model training and evaluation. This involves:
  - Generating and saving visualizations such as loss curves and mAP metrics.
  - Producing plots and statistics for review and reporting.

### `main.py`

- **Argument Parsing**: Handles command-line arguments and configurations for the experiment.
- **Model Initialization**: Initializes the chosen model (EfficientNet or SVM) based on provided arguments.
- **Training and Evaluation**: Manages the end-to-end process of:
  - Loading data.
  - Training the model.
  - Validating performance.
  - Saving results and metrics.

## Requirements
- Python 3.x
- PyTorch
- Numpy
- Matplotlib

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
