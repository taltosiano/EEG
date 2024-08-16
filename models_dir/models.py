import torch.nn as nn
import torch
from .HigherModels import *
from efficientnet_pytorch import EfficientNet
import torchvision
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json


class EffNetAttention(nn.Module):
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4):
        """
        Initializes the EffNetAttention model.

        Parameters:
        - label_dim (int): Number of output labels.
        - b (int): EfficientNet version (0 to 7).
        - pretrain (bool): Whether to use a pre-trained EfficientNet model.
        - head_num (int): Number of attention heads (0 for mean pooling, 1 for single-head attention, >1 for multi-head attention).
        """
        #This line calls the constructor of the parent class nn.Module
        super(EffNetAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
        # multi-head attention pooling
        if head_num > 1:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention = MHeadAttention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # single-head attention pooling
        elif head_num == 1:
            print('Model with single attention heads')
            self.attention = Attention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # mean pooling (no attention)
        elif head_num == 0:
            print('Model with mean pooling (NO Attention Heads)')
            self.attention = MeanPooling(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')

        self.avgpool = nn.AvgPool2d((1, 1))
        #remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out



class EEG_SVM_Classifier:
    def __init__(self, kernel='linear', C=1.0):
        """
        Initializes the EEG_SVM_Classifier.

        Parameters:
        - kernel (str): SVM kernel type (default is 'linear').
        - C (float): Regularization parameter for SVM (default is 1.0).
        """
        self.kernel = kernel
        self.C = C
        self.model = make_pipeline(StandardScaler(), SVC(kernel=self.kernel, C=self.C, random_state=42))
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, data_json):
        X = []
        y = []
        #data_list = [value for key, value in data_json_dict.items()]
        for i, (eeg_input, labels) in enumerate(train_loader):
            # if i>10:
            #     break
            features = np.concatenate(eeg_input.numpy())   # Concatenate EEG data features
            X.append(features)
            if labels[0][0] == 1.0:
                y.append('gap_element')
            else: y.append('plain_hit')
        X = np.array(X)
        X = X.reshape((X.shape[0], -1))
        y = self.label_encoder.fit_transform(y)
        return X,y

    def fit(self, train_data):
        """
        Trains the SVM model.

        Parameters:
        - train_data (dict): Dictionary containing training data and labels.
        """
        X_train, y_train = self.preprocess_data(train_data)
        self.model.fit(X_train, y_train)    # Fit the model

    def evaluate(self, data):
        X, y = self.preprocess_data(data)
        y_pred = self.model.predict(X)    # Predict using the model
        accuracy = accuracy_score(y, y_pred)    # Calculate accuracy
        return accuracy

    def predict(self, eeg_data):
        features = np.concatenate(eeg_data).reshape(1, -1)    # Prepare features
        label_encoded = self.model.predict(features)        # Predict label
        return self.label_encoder.inverse_transform(label_encoded)[0]    # Decode label
