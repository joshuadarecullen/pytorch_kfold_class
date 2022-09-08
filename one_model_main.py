from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german

import torch
from torch import nn
import torch.optim as optimiser
from torch.utils.data import Dataset

import numpy as np

import sys
sys.path.append('./kfold_pytorch')

from PytorchKFold import *

import models
from utils import *
from fairness_metrics import get_fairness_metrics as fair_mets


if __name__ == "__main__":

    # We define where's the bias in the features of our dataset.
    privileged_groups = [{'sex': 1, 'age': 1}]
    unprivileged_groups = [{'sex': 0, 'age':0}]

    np.random.seed(0)

    # load data in repsect to sex
    dataset_orig = load_preproc_data_german(['sex', 'age'])

    # We split between training and test set.
    train, test = dataset_orig.split([0.7], shuffle=True)

    # scales features and creates dataset, if no sensitive attribute processing ommit idxs
    train_dataset, test_dataset = get_dataset(train, test)

    # Initial parameters
    k, num_epochs, lr, batch_size, in_feature = 2, 100, 0.001, 10, train.features.shape[1]

    # set up model and loss 
    criterion = nn.BCELoss()
    model = models.LogisticRegression(in_feature, 1).double()
    optim = optimiser.Adam(model.parameters())

    model_processes = PytorchKFold(model, criterion, optim, train_dataset, k=k, lr=lr, batch_size=batch_size, epochs=num_epochs)

    '''
        in terms of changing hyper paramters, setting the self.optim every iteration and its self.reset_dict, same with model
    '''

    # print(model_train.kfold)
    k_avgs = model_processes.run_kfold()
    f_avgs, _ = fair_mets( k, train, model_processes.get_predictions(),
            unprivileged_groups, privileged_groups,
            model_processes.get_val_idxs(), flag=1)

    k_avgs.update(f_avgs)

    print(f'kfold avrgs:')
    for key, avg in k_avgs.items():
        print(f'{key}:{avg}')

    ''' 
    Computing final results
    get the models state dictionary stat_dict = torch.load(path_to_dict)
    Here simply uses model_processes.set_model(state_dict['model_state_dict']) to load best model
    Then, model_processes.predict(test_dataset)
    '''

