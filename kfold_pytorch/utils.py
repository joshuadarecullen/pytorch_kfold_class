import torch
from torch import nn
from torch.utils.data import Dataset


# Accumulates metrics i wanna know in a smart form
class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Simple pytorch custom data to store the sensitive feature separately
class MixenDataset(Dataset):

    def __init__(self, labels, features, z):
        self.labels = labels
        self.features = features
        self.z = z

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        z =self.z[idx]
        return feature, label, z


# Simple pytorch custom dataset
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


# This class normalises the dataset
class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x


# Converts a tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        super().__init__()

    def __call__(self, features, labels=None):

        # if labels.shape > 1
        features = torch.from_numpy(features)
        if len(labels) > 0:
            labels = torch.from_numpy(labels)
            # sum numpy arrays are 0 dimension, if so add a column dimension
            if len(labels.size()) == 1:
                labels = labels.unsqueeze(dim=1)

        return features, labels


def one_hot_encoder(one_hot_key, X):
    labels = torch.zeros(X.shape[0], len(one_hot_key.values()))
    for i, (age,sex) in enumerate(zip(X[:,0], X[:,1])):
        if age == 1 and sex == 1:
            labels[i] = torch.as_tensor(one_hot_key['1'])
        if age == 0 and sex == 0:
            labels[i] = torch.as_tensor(one_hot_key['2'])
        if age == 0 and sex == 1:
            labels[i] = torch.as_tensor(one_hot_key['3'])
        if age == 1 and sex == 0:
            labels[i] = torch.as_tensor(one_hot_key['4'])
    return labels


def scale_data(X_train, X_test):
    # Scale data
    scaler = TorchStandardScaler()
    scaler.fit(X_train)
    x_train = scaler.transform(X_train)
    x_test = scaler.transform(X_test)
    return x_train, x_test


def check_sensitive(z_idx, train, test):
    if len(z_idx) > 1:
        one_hot_key = {'1' : [1,0,0,0], '2' : [0,1,0,0], '3' : [0,0,1,0], '4' : [0,0,0,1]}
        y_train = one_hot_encoder(one_hot_key, train)
        y_test = one_hot_encoder(one_hot_key, test)

    return y_train, y_test


def convert_label(y_train, y_test, replace=0, value=2):
    y_train = torch.where(y_train==value, replace, y_train)
    y_test = torch.where(y_test==value, replace, y_test)
    return y_train, y_test


# Transfer into tensors then create dataset
def get_dataset(train, test, idx=None, z_idx=None, convert=True):

    if idx:
        To_Tensor = ToTensor()
        train_x, train_y = To_Tensor(train.features[:,idx], train.labels.ravel())
        test_x, test_y = To_Tensor(test.features[:,idx], test.labels.ravel())
        z_train, z_test = torch.tensor(train.features[:,z_idx]), torch.tensor(test.features[:,z_idx])

        # normalize data
        x_train, x_test = scale_data(train_x, test_x)

        # encoded adversarial label
        z_train, z_test = check_sensitive(z_idx, z_train, z_test)

        if convert:
            train_y, test_y = convert_label(train_y, test_y)

        # Create pytorch datasets
        train_dataset = MixenDataset(train_y, x_train, z_train)
        test_dataset = MixenDataset(test_y, x_test, z_test)

        return train_dataset, test_dataset

    else:
        To_Tensor = ToTensor()
        train_x, train_y = To_Tensor(train.features, train.labels.ravel())
        test_x, test_y = To_Tensor(test.features, test.labels.ravel())

        x_train, x_test = scale_data(train_x, test_x)

        if convert:
            train_y, test_y = convert_label(train_y, test_y)

        # Create pytorch datasets
        train_dataset = CustomDataset(train_y, x_train)
        test_dataset = CustomDataset(test_y, x_test)

        return train_dataset, test_dataset


def init_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

