import sys
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_german

import torch
from torch import nn
import torch.optim as optimiser

import numpy as np
import pandas as pd

sys.path.append('./kfold_pytorch')
from kfold_pytorch import *

import models
from pytorch_kfold_adversarial import *
from utils import *
from fairness_metrics import get_fairness_metrics as fair_mets

import plotting as plot

if __name__ == "__main__":

    # If class of either classifier or adversary is non binary or not
    non_binary =[False, True]

    # We define where's the bias in the features of our dataset.

    # binary
    # privileged_groups = [{'age':1}]
    # unprivileged_groups = [{ 'age':0}]

    # non binary
    privileged_groups = [{ 'age':1, 'sex': 1}]
    unprivileged_groups = [{ 'age':0, 'sex': 0}]

    # for repeating dataset split
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

    '''
    instatiate pytorch kfold class
    create a folder to store state of the model for use later, PATH is False by default
    '''
    model_processes = Pytorch_KFold_Adversary(cl_model, adv_model,
            cl_criterion, adv_criterion, train_dataset, cl_optim,
            adv_optim, k=k, epochs=num_epochs, batch_size=batch_size,
            pre_epoch=10, lr=lr, non_binary = non_binary, random_state=42,
            kf_shuffle=True)

    # set hyperparameter values
    adv_loss_weights = { 'alw1': 0.1,
                         'alw2': 0.2,
                         'alw3': 0.3,
                         'alw4': 0.4,
                         'alw5': 0.5,
                         'alw6': 0.6,
                         'alw7': 0.7,
                         'alw8': 0.8,
                         'alw9': 0.9,
                         'alw10': 1,
                         }

    # Keep track of averages of 5 folds for each lambda, fold, model history
    k_folds_data = {}

    # hyper loop
    for key, awl in adv_loss_weights.items():

        model_processes.reset_data()

        k_avgs = model_processes.run_kfold(adv_loss_weight=torch.tensor([awl]))

        f_avgs, _ = fair_mets(
                k,
                train,
                model_processes.get_predictions(),
                unprivileged_groups, privileged_groups,
                model_processes.get_val_idxs(),
                flag=1
                )

        k_avgs.update(f_avgs)

        print(f'kfold avrgs:')
        for key, avg in k_avgs.items():
            print(f'{key}:{avg}')

        k_folds_data[f'ALW:{awl}'] = k_avgs


    df_avrgs = pd.DataFrame(k_folds_data)

    plot.plot_multi_line(
            ['stat_par_diff', 'bal_acc', 'eq_opp_diff', 'avg_odds_diff'],
            list(adv_loss_weights.values()),
            df_avrgs,
            xlabel='Adversary Loss Weight',
            ylabel='Fairness Scores'
            )

    plot.plot_line(
            'cl_val_acc',
            list(adv_loss_weights.values()),
            df_avrgs,
            xlabel='Adversary Loss Weight'
            )

    '''
    Computing final results
    get the models state dictionary stat_dict = torch.load(path_to_dict)
    Here simply uses model_processes.set_model(state_dict['model_state_dict']) to load best model
    Then, model_processes.predict(test_dataset)
    or simply write your own function
    '''
