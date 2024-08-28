# Machine Learning-Driven Brain-Controlled Robotic One-Hand Exoskeleton
### `Authors`
- #### Tal Tosiano
- #### Noam David

## Overview

Our project, the **Machine Learning-Driven Brain-Controlled Robotic One-Hand Exoskeleton**, integrates advanced machine learning techniques with brain-computer interface (BCI) technology to provide intuitive control of a robotic exoskeleton using EEG signals. The system is composed of three primary components:
1. **Brain-Computer Interface (BCI)**: For EEG signal acquisition.
2. **Machine Learning Algorithms**: For signal processing and classification.
3. **Robotic Exoskeleton**: For providing physical assistance.

The project leverages deep neural networks and adaptive algorithms to interpret EEG signals in real time, enabling users to control the exoskeleton with minimal cognitive effort.

## Data Acquisition
### Emotiv EPOC EEG Headset
We utilized the **Emotiv EPOC** EEG headset for data acquisition. This wireless device offers a cost-effective solution for recording brain activity with 14 active electrodes placed around the head. The EEG signals are sampled at 128Hz and undergo bandpass filtering (2-30Hz) to isolate different frequency bands (delta, theta, alpha, beta). The Fast Fourier Transform (FFT) is then applied to calculate the signal power at each frequency.

## Project Structure & Description

- #### `data_prep.py`:
  Contains the `EEGDataset` class for loading and processing EEG data. It manages the data pipeline, including:
  - Reading data from JSON files.
  - Applying necessary transformations.
  - Preparing data batches for training and evaluation.
- #### `models_dir/models.py`: 
  Defines model architectures, including `CNN` and `Attention`.
- #### `traintest.py`: 
  Implements training (`train`) and validation (`validate`) functions for the EEG model.
- **`train`**: This function handles the training of the EEG model, including:
  - The training loop with forward and backward passes.
  - Optimization and loss computation.
  - Saving model checkpoints and tracking training progress.
  
- **`validate`**: This function evaluates the model on validation or test datasets. It computes:
  - Metrics such as mAP (mean Average Precision) and AUC (Area Under the Curve).
  - Saves prediction results for further analysis.
    
- #### `data_prep.py`: 
  Contains various utility functions for preparing and preprocessing EEG data. This includes:
  - Data normalization.
  - Splitting data into training, validation, and testing sets.
  - Other necessary preprocessing steps before model training.
- #### `DataAnalysis.py`: 
  Provides functions for analyzing the results of model training and evaluation. This involves:
  - Generating and saving visualizations such as loss curves and mAP metrics.
  - Producing plots and statistics for review and reporting.
- #### `main.py`:
  - **Argument Parsing**: Handles command-line arguments and configurations for the experiment.
  - **Model Initialization**: Initializes the chosen model (EfficientNet or SVM) based on provided arguments.
  - **Training and Evaluation**: Manages the end-to-end process of:
    - Loading data.
    - Training the model.
    - Validating performance.
    - Saving results and metrics.
- #### `meas.py`: 
  Contains a snippet for calculating and printing eigenvalues and eigenvectors of a predefined matrix.
- #### `sample.py`: 
  Provides functionality to evaluate a sample EEG signal either from a test set or an external CSV file. It uses a pre-trained model    to classify the sample as either a button press (plain_hit) or no button press (gap_element).
- #### `parse_code.py`: 
  Defines and parses command-line arguments for training and evaluation of the EEG model.


## Methodology

### Data Acquisition
- **Emotiv EPOC**: We use the Emotiv EPOC headset, a cost-effective option for recording brain activity. It features 14 active electrodes, providing EEG signal measurements with a sampling rate of 128Hz. The recorded data undergoes bandpass filtering (2-30Hz) and Fast Fourier Transform (FFT) to calculate the signal power at each frequency.

### Data Preprocessing
- **Event Alignment**: We synchronized the EEG data with event markers by aligning the duration and timestamps of events from the marker data with the corresponding EEG measurements.
- **Sample Balancing**: We identified imbalances in the dataset, particularly with the "pattern" label, and focused on "gap_element" and "plain_hit" labels for further processing.
- **Dimensionality Reduction**: We standardized the time dimension of each sample to 62 rows (~0.6 seconds), cutting large samples and zero-padding shorter ones.

### Feature Extraction
- **Short-Time Fourier Transform (STFT)**: Input data undergoes STFT processing, resulting in [62x64] matrices representing time-frequency features. We used dominant EEG channels for further analysis.
- **Attention Mechanism**: We implemented a Multi-Head Attention mechanism to improve the representation of sequential data, leveraging the relationship between different parts of the EEG signals.

### Model Architecture
- **EfficientNet**: The backbone of our model is the EfficientNet CNN, which processes the STFT output. We employed transfer learning, using pre-trained weights from ImageNet to enhance feature extraction.
- **Embedding**: After the CNN, we used an embedding process to improve the representation of the features before feeding them into the attention mode
- **Multi-Head Attention**: Following the CNN, we applied a 4-head Multi-Head Attention mechanism to refine the extracted features, improving classification accuracy.
- **Training**: The model was trained using Cross Entropy loss, optimized with Adam, and a learning rate scheduler to avoid overfitting.

## The results of the trained model
<img width="241" alt="image" src="https://github.com/user-attachments/assets/5eb435c1-dfcc-49b0-8780-368c8cdfc7c5">
<img width="238" alt="image" src="https://github.com/user-attachments/assets/50136bf6-ea1c-45ff-87ba-f317eae3aea9">

##### On the left-The plot of loss on train and validation set vs epochs.
##### On the right-The mAP grade on our validation set every epoch.


## Usage
1. **Train the Model**:
To train the model, you need to run the main file. This will create an exp directory with the trained model and other outputs. Additionally, it will return the accuracy percentage of the model for correct classification.
   ```bash
   python main.py --exp-dir <path_to_exp_directory>
   
2. **Test Individual Samples**:
After training, you can use the sample.py script to test individual samples.
   ```bash
   python sample.py --own_idx [True] —samp_idx [Number of sample from train.json]

![image](https://github.com/user-attachments/assets/eef7d61c-e49a-4e6a-a31f-44b58d865d22)

