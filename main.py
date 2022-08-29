from PytorchKFold import *

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.optim as optimiser
from torch.utils.data import Dataset

import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Simple pytorch custom data to structure features and labels
class CustomDataset(Dataset):

    def __init__(self, labels, features):
        self.labels = labels
        self.features = features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


if __name__ == "__main__":

    #STEP 2: We define where's the bias in the features of our dataset.
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    np.random.seed(0)

    # load data in repsect to sex
    dataset_orig = load_preproc_data_german(['sex'])

    #STEP 3: We split between training and test set.
    train, test = dataset_orig.split([0.7], shuffle=True)

    #Normalize the dataset, both train and test.
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(train.features)
    y_train = train.labels.ravel()
    X_test = scale_orig.transform(test.features)
    y_test = test.labels.ravel()

    # Transfer into tensors
    X_train_t, X_test_t = torch.tensor(X_train, dtype=torch.float64), torch.tensor(X_test, dtype=torch.float64)
    y_train_labels = torch.tensor(y_train.reshape(-1,1), dtype=torch.float64)
    y_test_labels = torch.tensor(y_test.reshape(-1,1), dtype=torch.float64)

    # create pytorch dataset
    train_dataset = CustomDataset(y_train_labels, X_train_t)
    test_dataset = CustomDataset(y_test_labels, X_test_t)

    # Initial parameters
    k, num_epochs, lr, batch_size, in_feature = 5, 100, 0.001, 10, X_train_t.shape[1]

    # set up model and loss 
    criterion = nn.BCELoss()
    model = LogisticRegression(in_feature, 1).double()
    optim = optimiser.Adam(model.parameters())

    model_train = PytorchKFold(model, criterion, train_dataset, optim, k=k, lr=lr, batch_size=batch_size, epochs=num_epochs)

    # print(model_train.kfold)
    k_avrgs = model_train.run_kfold()
