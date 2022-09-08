import sys
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german

import torch
from torch import nn
import torch.optim as optimiser
import numpy as np

sys.path.append('./kfold_pytorch')
from kfold_pytorch import *

import models
from pytorch_kfold_adversarial import *
from utils import *
from fairness_metrics import get_fairness_metrics as fair_mets


if __name__ == "__main__":

    # If class of either classifier or adversary is non binary or not
    non_binary =[False, True]

    # We define where's the bias in the features of our dataset.
    # privileged_groups = [{'age':1}]
    # unprivileged_groups = [{ 'age':0}]

    privileged_groups = [{ 'age':1, 'sex': 1}]
    unprivileged_groups = [{ 'age':0, 'sex': 0}]

    np.random.seed(0)

    # load data in repsect to sex
    dataset_orig = load_preproc_data_german(['age','sex'])

    # We split between training and test set.
    train, test = dataset_orig.split([0.7], shuffle=True)

    # position of sensitive attributes
    z_pos = [0,1]

    # get idx of sensitive feature to remove from data
    idx = [x for x in range(train.features.shape[1]) if x not in z_pos]
    z_idx = [x for x in range(train.features.shape[1]) if x in z_pos]

    # scales features and creates dataset, if no sensitive attribute processing ommit idxs
    train_dataset, test_dataset = get_dataset(train, test, idx=idx, z_idx=z_idx)

    # Initial parameters
    k, num_epochs, lr, batch_size, in_feature = 2, 100, 0.001, 10, len(idx)

    # set up loss, model and optimiser for both models
    cl_criterion = nn.CrossEntropyLoss() if non_binary[0] else nn.BCELoss()
    cl_model = models.LogisticRegression(in_feature, 1).double()
    cl_optim = optimiser.Adam(cl_model.parameters())

    adv_criterion = nn.CrossEntropyLoss() if non_binary[1] else nn.BCELoss()
    adv_model = models.Adversarial_LogisticRegression(2, 4).double()
    adv_optim = optimiser.Adam(adv_model.parameters())

    # instatiate pytorch kfold class
    model_processes = Pytorch_KFold_Adversary(cl_model, adv_model,
            cl_criterion, adv_criterion, train_dataset, cl_optim,
            adv_optim, k=k, epochs=num_epochs, batch_size=batch_size,
            pre_epoch=10, lr=lr, non_binary = non_binary, random_state=42,
            kf_shuffle=True, PATH=None)

    # set hyperparameter values
    awls = torch.arange(0.1,1,0.1)

    # hyper loop
    for awl in awls:

        model_train.reset_data()

        k_avgs = model_processes.run_kfold(adv_loss_weight=awl)

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
    or simply write your own function
    '''
