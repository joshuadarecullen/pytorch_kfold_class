from sklearn.model_selection import KFold
import sys
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from utils import Accumulator, init_weights

class PytorchKFold:

    def __init__(self, model, criterion, optim, dataset, k=10, epochs=100, batch_size=32,
            lr=0.001, random_state=42, kf_shuffle=True, PATH=None,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        # torch.manual_seed(random_state)
        self.model = model.to(device)
        self.optim = optim
        self.criterion = criterion
        self.dataset = dataset

        # save model, optimiser and hyperparameters
        self.PATH = PATH

        # Hyper parameters
        self.k = k
        self.epochs = epochs
        self.batch_size = batch_size
        # self.wd = wd
        # self.lr = lr

        # reproducability states and shuffle states
        self.random_state = random_state
        self.kf_shuffle = kf_shuffle

        # device training done on
        self.device = device
        self.reset_optim = optim.state_dict()

        # split data into train and validation
        self.kfold = KFold(n_splits = self.k, shuffle = True, random_state = self.random_state)

        # Data collected from all folds
        self.all_folds_history = {}
        self.all_folds_predictions = {}
        self.all_folds_probabilities = {}
        self.all_folds_true = {}
        self.all_folds_val = {}
        self.all_val_idx = {}

        self.reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
        self.astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


    def run_kfold(self):

        # keep track of performance and fairness metrics over epochs
        avr_metrics = Accumulator(4)

        # main kfold loop
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.dataset)):
 
            print(f'Fold {fold+1}')

            # set up model and place on gpu/cpu
            self.model.apply(init_weights)
            self.optim.load_state_dict(self.reset_optim)

            # get subset of training data given the kfold indices 
            train_sub, val_sub = self.get_kfold_data(train_idx, val_idx)

            # set the dataloaders
            train_loader = DataLoader(train_sub, batch_size=self.batch_size,
                    shuffle=True, num_workers=8)
            val_loader = DataLoader(val_sub, batch_size=self.batch_size, 
                    shuffle=False)

            # Main training loop
            train_loss, train_acc, val_loss, val_acc, train_history = self.train(train_loader, val_loader)

            if self.PATH:
                # Save the state of the models parameters
                torch.save({
                        'epoch': self.epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        'train_loss_history': train_loss,
                        }, self.PATH + f'fold-{fold+1}.pth')

            # Final predictions using validation set for fairness metrics
            predictions, probs, true, accuracy = self.predict(val_loader)

            avr_metrics.add(train_loss,
                            train_acc,
                            val_loss,
                            val_acc)

            # update performance trackers
            self.all_folds_history[f'Fold:{fold+1}'] = train_history
            self.all_folds_predictions[f'Fold:{fold+1}'] = predictions
            self.all_folds_true[f'fold:{fold+1}'] = true
            self.all_folds_probabilities[f'fold:{fold+1}'] = probs
            self.all_folds_val[f'fold:{fold+1}'] = accuracy
            self.all_val_idx[f'fold:{fold+1}'] = val_idx


            # print loss and accuracy metrics
            print(f'Train loss: {train_loss}')
            print(f'Validation accuracy: {accuracy}')

        # Calculate Averages across the folds
        self.kfolds_avergs = {'num_folds': self.k,
                       'train_loss': [avr_metrics[0]/self.k],
                       'train_acc':[avr_metrics[1]/self.k],
                       'val_loss': [avr_metrics[2]/self.k],
                       'val_acc': [avr_metrics[3]/self.k],
                       }

        return self.kfolds_avergs


    # Training loop
    def train_epoch(self, data_iter):

        # collect epoch data
        metric = Accumulator(4)
        # Put model in train mode
        self.model.train()

        for features, labels in data_iter:
            features, labels = features.to(self.device), labels.to(self.device)

            # Foward pass
            y_hat = self.model(features)

            # Calculate loss
            loss = self.criterion(y_hat, labels)

            # Clear gradients
            self.optim.zero_grad()

            # Calculate loss w.r.t params
            loss.mean().backward()

            # perform gradient descent
            self.optim.step()

            metric.add(float(loss.sum()), self.accuracy(y_hat,labels), labels.numel(), 1)

        # return training loss and training accuracy
        return metric[0]/metric[3], (metric[1]/ metric[2]) * 100


    # Task one training loop
    def train(self, train_loader, val_loader):

        # Store the folds performance
        history = {'train_loss': [], 'train_acc':[], 'test_loss': [], 'test_acc': []}

        metrics = Accumulator(4)

        # Training loop
        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(train_loader)
            test_metrics = self.evaluate_accuracy(val_loader)

            # keep full history
            history['train_loss'].append(train_metrics[0])
            history['train_acc'].append(train_metrics[1])
            history['test_loss'].append(test_metrics[0])
            history['test_acc'].append(test_metrics[1])

            # Track epoch loss and acc for both train and val data
            metrics.add(train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1])

        return (metrics[0]/self.epochs, metrics[1]/self.epochs, metrics[2]/self.epochs, metrics[3]/self.epochs, history)


    # Predict given a model and dataset for datasets with only label and features
    def predict(self, data_iter):
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()  # Set the model to evaluation mode

        y_pred = []
        y_prob = []
        y_true = []

        metric = Accumulator(2)  # No. of correct predictions, no. of predictions
        with torch.no_grad():
            for X,y in data_iter:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                out = output.detach().clone()
                y_probs = out.data.cpu().numpy() #send to cpu to save
                y_prob.extend(y_probs)

                metric.add(self.accuracy(output, y), y.numel())

                y_hat = (torch.round(output)).data.cpu().numpy() #send to cpu to save
                y_pred.extend(y_hat) # Save prediction
                labels = y.data.cpu().numpy()
                y_true.extend(labels)

        ypred = np.asarray(y_pred)
        ytrue = np.asarray(y_true)
        yprob = np.asarray(y_prob)

        # Return predictions, model prob output, truth and accuracy
        return  ypred, yprob, ytrue, (metric[0] / metric[1]) * 100


    # grab the subet from the data given kfold idxs
    def get_kfold_data(self, train_idx, val_idx):
        train_sub = Subset(self.dataset, train_idx) 
        val_sub = Subset(self.dataset, val_idx) 
        return train_sub, val_sub


    # evaluate the accuracy of the model
    def accuracy(self, y_hat, y):
        y_hat = torch.round(y_hat)
        cmp = self.astype(y_hat, y.dtype) == y # convert y_hat into a data type that matches y 
        correct = float(self.reduce_sum(self.astype(cmp, y.dtype))) # Taking the sum yields the number of correct predictions.
        return correct


    # used primarily in the first task for both dataset for checking validation accuracy through batches
    def evaluate_accuracy(self, data_iter):

        # Compute the accuracy for a model on a dataset.
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()  # Set the model to evaluation mode

        metric = Accumulator(4)  # No. of correct predictions, no. of predictions
        with torch.no_grad():
            for X, y in data_iter:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                metric.add(float(loss.sum()), self.accuracy(output,y), y.numel(), 1)

        # return test loss and  accuracy
        return metric[0] / metric[3], (metric[1] / metric[2])*100


    def get_folds_data():
        return (self.all_folds_history, self.all_folds_predictions, self.all_folds_probabilities, self.all_folds_true, self.folds_val)

    def get_val_idxs(self):
        return self.all_val_idx

    def get_predictions(self):
        return self.all_folds_predictions

    def set_model(self, state_dict):
        self.model.load_state_dict(state_dict)
