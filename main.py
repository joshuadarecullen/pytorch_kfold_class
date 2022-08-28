from PytorchKFold import *

from aif360.datasets import AdultDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from sklearn.preprocessing import StandardScaler
import torch.optim as optimiser
import torch.Dataset as Dataset


### Collection of utils used throughout the code
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


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)     
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


if __name__ == "__main__":

    #STEP 2: We define where's the bias in the features of our dataset.
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    np.random.seed(0)

    # load data in repsect to sex
    dataset_orig = load_preproc_data_adult(['sex'])

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

    train_dataset = Dataset(y_train_labels, X_train_t)
    test_dataset = Dataset(y_test_labels, X_test_t)

    # Initial parameters
    k, num_epochs, lr, batch_size, in_feature = 5, 100, 0.001, 64, 18
    weight_decay = {'wd2': 1000,'wd3': 100,'wd4': 10,'wd5': 1 ,'wd6': 0.1, 'wd7': 0.01,'wd8': 0.001}

    # set up model and loss 
    criterion = nn.BCELoss()
    model = LogisticalRegression(X_train_t.shape[1]).double()
    optim = optimiser.Adam(model.parameters())

    kfold = PytorchKFold(model, criterion, train_dataset, optim, k=1)
