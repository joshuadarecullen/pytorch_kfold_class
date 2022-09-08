from sklearn.model_selection import KFold
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from utils import Accumulator, init_weights


class Pytorch_KFold_Adversary:
    def __init__(self, cl_model, adv_model,
            cl_criterion, adv_criterion, dataset,
            cl_optim, adv_optim, k=10, epochs=100,
            batch_size=32, pre_epoch=1,
            adv_loss_weight=None,
            lr=0.001, random_state=42, kf_shuffle=True,
            PATH=None, measure=None, non_binary = [False,False],
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        # torch.manual_seed(random_state)
        self.cl_model = cl_model.to(device)
        self.adv_model = adv_model.to(device)
        self.cl_optim = cl_optim
        self.adv_optim = adv_optim
        self.cl_criterion = cl_criterion
        self.adv_criterion = adv_criterion
        self.dataset = dataset

        # save model, optimiser and hyperparameters
        self.PATH = PATH

        # Hyper parameters
        self.k = k
        self.epochs = epochs
        self.batch_size = batch_size
        # self.wd = wd
        self.lr = lr
        self.adv_loss_weight = adv_loss_weight.to(device) if adv_loss_weight else None

        # reproducability states and shuffle states
        self.random_state = random_state
        self.kf_shuffle = kf_shuffle

        # device training done on
        self.device = device
        self.reset_optim_cl = cl_optim.state_dict()
        self.reset_optim_adv = adv_optim.state_dict()
        self.measure = measure
        self.non_binary = non_binary

        # split data into train and validation
        self.kfold = KFold(n_splits = self.k, shuffle = True, random_state = self.random_state)

        # Set-up functions
        self.accuracy = self.accuracyCE if self.non_binary[0] else self.accuracyBCE
        self.get_adv_accuracy = self.get_adv_accuracyCE if self.non_binary[1] else self.get_adv_accuracyBCE
        self.train_func =  self.train_adv_eq_opp if self.measure else self.train_adv_proj_eq_odds
        self.predict = self.predict_adv_equal_opp if self.measure else self.predict_adv

        # Data collected from all folds
        self.all_folds_history = {}
        self.all_folds_predictions = {}
        self.all_folds_probabilities = {}
        self.all_folds_true = {}
        self.all_folds_val = {}
        self.all_folds_adv_probabilities = {}
        self.all_val_idx ={}

        self.reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
        self.astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


    def run_kfold(self, adv_loss_weight=None):
        print(adv_loss_weight)

        if adv_loss_weight:
            self.adv_loss_weight = adv_loss_weight.to(self.device)

        # keep track of performance and fairness metrics over epochs
        avr_metrics = Accumulator(8)

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.dataset)):

            print(f'Fold {fold+1}, adversary loss weight: {float(self.adv_loss_weight)}')

            # get the data for the kfold
            train_sub, val_sub = self.get_kfold_data(train_idx, val_idx)

            # load the train data 
            train_loader = DataLoader(train_sub, batch_size=self.batch_size,
                    shuffle=True, num_workers=8)

            # load the test data
            val_loader = DataLoader(val_sub, batch_size=self.batch_size,
                    shuffle=False, num_workers=8)

            # Set up model
            self.cl_model.apply(init_weights)
            self.adv_model.apply(init_weights)
            self.cl_optim.load_state_dict(self.reset_optim_cl)
            self.adv_optim.load_state_dict(self.reset_optim_adv)


            # main train function call, handles pre training as well.
            ptclf_acc, ptadv_acc, cl_train_loss, adv_train_loss, adv_train_acc, cl_train_acc, history = self.train_func(train_loader)

            predictions, probs, true, sprob, cl_accuracy, adv_acc = self.predict(val_loader)

            # Save the state of the models parameters
            if self.PATH:
                torch.save(self.cl_model.state_dict(), self.PATH + f'fold-{fold+1}-classifier.pth')
                torch.save(self.adv_model.state_dict(), self.PATH + f'fold-{fold+1}-adv.pth')

            # accumulate the fold metrics
            avr_metrics.add(cl_train_loss,
                            cl_train_acc,
                            adv_train_loss,
                            adv_train_acc,
                            ptclf_acc,
                            ptadv_acc,
                            cl_accuracy,
                            adv_acc)

            # update performance trackers
            self.all_folds_predictions[f'Fold{fold+1}'] = predictions
            self.all_folds_true[f'fold{fold+1}'] = true
            self.all_folds_probabilities[f'fold{fold+1}'] = probs
            self.all_folds_adv_probabilities[f'fold{fold+1}'] = sprob
            self.all_folds_history[f'Fold{fold+1}'] = history
            self.all_folds_val[f'fold{fold+1}'] = cl_accuracy
            self.all_val_idx[f'fold{fold+1}'] = val_idx

           # Calculate Averages across the folds
        self.kfolds_avergs = {'num_folds': self.k,
                       'cl_train_loss_debias': [avr_metrics[0]/self.k],
                       'cl_train_acc_debias':[avr_metrics[1]/self.k],
                       'pre_cl_train_acc':[avr_metrics[4]/self.k],
                       'adv_train_loss_debias': [avr_metrics[2]/self.k],
                       'adv_train_acc_debias': [avr_metrics[3]/self.k],
                       'pre_adv_train_acc': [avr_metrics[5]/self.k],
                       'cl_val_acc': [avr_metrics[6]/self.k],
                       'adv_val_acc': [avr_metrics[7]/self.k],
                       }

        return self.kfolds_avergs


    # Main training loop for adversarial debiasing with the projection term
    def train_adv_proj_eq_odds(self, dataset, pre_epoch=5):

        self.adv_model.train()
        self.cl_model.train()

        # pre-train both models 
        ptclf_loss, ptclf_acc, ptclf_history = self.pre_train_class(dataset, N_CLF_EPOCHS=pre_epoch)
        ptadv_loss, ptadv_acc, ptadv_cl_acc, ptadv_history = self.pre_train_ad(dataset, N_ADV_EPOCHS=pre_epoch)

        history = {'adv_train_loss_debias': [],'adv_train_acc_debias': [], 'cl_train_loss_debias': [], 'cl_train_acc_debias': []}
        epoch_met = Accumulator(5)

        # Main trainig loop, the adversary is trained on whole dataset to hold the advantage
        for epoch in range(1, self.epochs):
            metric = Accumulator(6)

            self.adv_model.train()
            self.cl_model.train()

            for features, labels, sensitive in dataset:
                features, labels, sensitive = features.to(self.device), labels.to(self.device), sensitive.to(self.device)

                self.cl_optim.zero_grad()
                self.adv_optim.zero_grad()

                # classifier prediction
                cl_pred = self.cl_model(features)

                # adversary sees both the ground truth and the classifiers prediction
                z = torch.cat((cl_pred, labels), 1)
                protect_pred = self.adv_model(z)

                pred_loss = self.cl_criterion(cl_pred, labels)
                protect_loss = self.adv_criterion(protect_pred, sensitive)

                protect_loss.backward(retain_graph=True) # keep graph to use
                protect_grad = {name: param.grad.clone() for name, param in self.cl_model.named_parameters()} # grab gradient

                self.adv_optim.step()

                self.cl_optim.zero_grad()
                pred_loss.backward()

                with torch.no_grad():
                    for name, param in self.cl_model.named_parameters():
                        unit_protect = self.normalize(protect_grad[name]) # grabbing the norm of the classifiers gradient
                        param.grad -= ((param.grad * unit_protect) * unit_protect.sum()) # projection
                        param.grad -= self.adv_loss_weight * protect_grad[name] # hurting the adversary

                self.cl_optim.step()

                metric.add(float(pred_loss), float(protect_loss), self.get_adv_accuracy(protect_pred, sensitive), self.accuracy(cl_pred,labels), labels.numel(), 1)

            history['adv_train_loss_debias'].append(metric[1]/metric[5])
            history['adv_train_acc_debias'].append(metric[2]/metric[4])
            history['cl_train_acc_debias'].append(metric[3]/metric[4])
            history['cl_train_loss_debias'].append(metric[0]/metric[5])

            # adv_train_loss, adv accurac, cl acc, cl loss - debiased
            epoch_met.add(metric[1]/metric[5], metric[2]/metric[4], metric[3]/metric[4], metric[0]/metric[5])

        print(f'classifier final train loss--> {epoch_met[3]/self.epochs}')
        # print(f'classifier final train acc--> {epoch_met[2]/epochs}')
        print(f'Adversary final train loss--> {epoch_met[0]/self.epochs}')
        # print(f'Adversary final train accuracy--> {epoch_met[1]/epochs}')

        return (ptclf_acc, ptadv_acc, epoch_met[3]/self.epochs, epoch_met[0]/self.epochs, epoch_met[1]/self.epochs, epoch_met[2]/self.epochs, history)


    # Training without projection and adversary on y = 1
    def train_adv_eq_opp(self, train_loader, pre_epoch=5):

        self.cl_model.train()
        self.adv_model.train()

        # return the pretraining metrics
        ptclf_loss, ptclf_acc, ptclf_history = self.pre_train_class(train_loader, N_CLF_EPOCHS=pre_epoch)
        ptadv_loss, ptadv_acc, ptadv_cl_acc, ptadv_history = self.pre_train_ad_eq_opp(train_loader, N_ADV_EPOCHS=pre_epoch)
        ptclf_stats = {'cl_loss': ptclf_loss, 'cl_acc': ptclf_acc, 'history': ptclf_history}
        ptadv_stats = {'adv_loss': ptadv_loss, 'adv_acc': ptadv_acc, 'cl_acc': ptadv_cl_acc ,'history': ptadv_history}

        # collect epoch data
        epoch_met = Accumulator(7)

        # Store all the history for both models, their pre train and second pahse training
        history = {'adv_loss': [], 'cl_acc_adv': [],
                'adv_acc': [], 'adv_acc_debias': [], 'cl_model_debias': [],
                'adv_loss_debias': [],'cl_loss_debias': []}

        # Main trainig loop for and the adversary is trained on whole dataset to hold the advantage
        for epoch in range(1, self.epochs):
            adv_metric = Accumulator(6)
            for features, labels, sensitive in train_loader:

                features, labels, sensitive = features.to(self.device), labels.to(self.device), sensitive.to(self.device)

                idx = self.get_idx(labels)

                # get next batch if no y = 1 labels
                if len(idx) == 0:
                    continue

                self.adv_model.zero_grad() # no gradient accumulation

                p_y = self.cl_model(features).detach() # save memory
                ad_input = torch.cat((p_y, labels), 1) # rework input for adversary to include labels
                p_z = self.adv_model(ad_input) # predict adversary

                # calculate loss and optimise adversary network
                loss_adv = self.adv_criterion(p_z[idx], sensitive[idx])
                loss_adv.backward()
                self.adv_optim.step()

                # collect metrics
                adv_metric.add(float(loss_adv), self.accuracy(p_y,labels), self.get_adv_accuracy(p_z[idx], sensitive[idx]), labels.numel(), labels[idx].numel(), 1)


            # keep full history
            history['adv_loss'].append(adv_metric[0]/adv_metric[5])
            history['cl_acc_adv'].append((adv_metric[1]/adv_metric[3])*100)
            history['adv_acc'].append((adv_metric[2]/adv_metric[4]*100))

            # ugly way of grabbing one batch in the network 
            for features, labels, sensitive in train_loader:
                features, labels, sensitive = features.to(self.device), labels.to(self.device), sensitive.to(self.device)
                pass

            idx = self.get_idx(labels)

            # not the prettiest but there is a chance no y = 1 are in the batch and causes an error in metric calcs
            while len(idx) == 0:
                for features, labels, sensitive in train_loader:
                    features, labels, sensitive = features.to(self.device), labels.to(self.device), sensitive.to(self.device)
                    pass
                idx = self.get_idx(labels)

            # dont collect gradients
            self.cl_model.zero_grad()

            #predict probability
            p_y = self.cl_model(features)

            # for equal odds only the adversary sees both cl ouput and ground truth
            ad_input = torch.cat((p_y, labels), 1)
            p_z = self.adv_model(ad_input)

            # calculate adversary loss, only y = 1 labels are examined
            loss_adv = self.adv_criterion(p_z[idx], sensitive[idx]) * self.adv_loss_weight

            # calculate loss of the classifier with the negation from the adversary
            cl_loss = self.cl_criterion(p_y, labels) - (adv_criterion(self.adv_model(ad_input[idx]),sensitive[idx]) * self.adv_loss_weight)
            cl_loss.backward()

            # optimise after updates to the weights of the classifier
            self.cl_optim.step()

            # keep full history
            history['adv_acc_debias'].append((get_adv_accuracy(p_z[idx], sensitive[idx])/labels[idx].numel())*100)
            history['cl_model_debias'].append((accuracy(p_y, labels)/labels.numel())*100)
            history['adv_loss_debias'].append(float(loss_adv))
            history['cl_loss_debias'].append(float(cl_loss))

            # tracking all metrics for both models
            epoch_met.add(adv_metric[0]/adv_metric[5],
                    (adv_metric[1]/adv_metric[3])*100,
                    (adv_metric[2]/adv_metric[4])*100,
                    float(loss_adv),
                    float(cl_loss),
                    (self.accuracy(p_y, labels)/labels.numel())*100,
                    (self.get_adv_accuracy(p_z[idx], sensitive[idx])/labels[idx].numel()*100))


        # print(f'classifier final train loss--> {epoch_met[4]/epochs}')
        # print(f'classifier final train acc--> {epoch_met[5]/epochs}')
        # print(f'Adversary final train loss--> {epoch_met[3]/epochs}')
        # print(f'Adversary final train accuracy--> {epoch_met[6]/epochs}')

        # return training loss and training accuracy
        return ptclf_acc, ptadv_acc, epoch_met[4]/self.epochs, epoch_met[3]/self.epochs, epoch_met[6]/self.epochs, epoch_met[5]/self.epochs, history



    # Pretrain the classifier 
    def pre_train_class(self, data_iter, N_CLF_EPOCHS=5):

        history = {'train_loss': [],'train_acc': []} 
        epoch_met = Accumulator(4)
         # very small number in pre training, just enough to stabalise

        for epoch in range(N_CLF_EPOCHS):

            metric = Accumulator(4)

            for features, labels, _ in data_iter:
                features, labels = features.to(self.device), labels.to(self.device)

                self.cl_model.zero_grad() # no gradient accumulation
                p_y = self.cl_model(features)
                loss = self.cl_criterion(p_y, labels)
                loss.backward()
                self.cl_optim.step()

                metric.add(float(loss.sum()), self.accuracy(p_y,labels), labels.numel(), 1)

            # keep full history
            history['train_loss'].append(metric[0]/metric[3])
            history['train_acc'].append((metric[1]/metric[2])*100)

            epoch_met.add(metric[0] / metric[3], (metric[1]/ metric[2]) * 100)

        return epoch_met[0]/N_CLF_EPOCHS, epoch_met[1]/N_CLF_EPOCHS, history


    # pre train the adversary more to give it an advantage
    def pre_train_ad(self, data_iter, N_ADV_EPOCHS=5):

        epoch_met = Accumulator(3)
        history = {'adv_train_loss': [],'adv_train_acc': [], 'cl_train_acc': []}

        # main training epoch loop
        for epoch in range(N_ADV_EPOCHS):

            metric = Accumulator(5)

            for features, labels, sensitive in data_iter:
                # collect epoch data
                features, sensitive, labels = features.to(self.device), sensitive.to(self.device), labels.to(self.device)

                # dont collect gradients for ad
                self.adv_model.zero_grad()
                p_y = self.cl_model(features).detach()

                # rework input into adversary to include ground truth
                ad_input = torch.cat((p_y, labels), 1)
                p_z = self.adv_model(ad_input)

                # calculate loss of the adversary
                loss = self.adv_criterion(p_z,sensitive)
                loss.backward()

                # optimise
                self.adv_optim.step()

                # track epoch data
                metric.add(float(loss), self.get_adv_accuracy(p_z,sensitive), self.accuracy(p_y, labels), labels.numel(), 1)

            # keep full history
            history['adv_train_loss'].append(metric[0]/metric[4])
            history['adv_train_acc'].append((metric[1]/metric[3])*100)
            history['cl_train_acc'].append((metric[2]/metric[3])*100)

            # track epoch metrics
            # avergae loss. adversay training and classifier accucary
            epoch_met.add(metric[0]/metric[4], (metric[1]/metric[3])*100, (metric[2]/metric[3])*100)

        return (epoch_met[0]/N_ADV_EPOCHS, epoch_met[1]/N_ADV_EPOCHS, epoch_met[2]/N_ADV_EPOCHS, history)


    # pre train the adversary more to give it an advantage
    def pre_train_ad_eq_opp(self, data_iter, N_ADV_EPOCHS=5):

        epoch_met = Accumulator(3)
        history = {'adv_train_loss': [],'adv_train_acc': [], 'cl_train_acc': []}

        # main training epoch loop
        for epoch in range(N_ADV_EPOCHS):

            metric = Accumulator(6)

            for features, labels, sensitive in data_iter:
                # collect epoch data
                features, sensitive, labels = features.to(self.device), sensitive.to(self.device), labels.to(self.device)

                # grab only when y = 1
                idx = self.get_idx(labels)

                if len(idx) == 0:
                    continue

                # dont collect gradients for ad
                self.adv_model.zero_grad()
                p_y = self.cl_model(features).detach()


                # rework input into adversary to include ground truth
                ad_input = torch.cat((p_y, labels), 1)
                p_z = self.adv_model(ad_input)

                # calculate reworked ad loss, only with respects to minimising the weights to y = 1
                loss = self.adv_criterion(p_z[idx],sensitive[idx]) # include #[idx], same for get_adv_acc below and labels
                loss.backward()

                # optimise
                self.adv_optim.step()

                # track epoch data
                metric.add(loss.sum(), self.get_adv_accuracy(p_z[idx],sensitive[idx]), self.accuracy(p_y, labels), labels.numel(), labels[idx].numel(), 1)

            # keep full history
            history['adv_train_loss'].append(metric[0]/metric[5])
            history['adv_train_acc'].append((metric[1]/metric[4])*100)
            history['cl_train_acc'].append((metric[2]/metric[3])*100)

            # track epoch metrics
            # avergae loss. adversay training and classifier accucary
            epoch_met.add(metric[0]/metric[5], (metric[1]/metric[4])*100, (metric[2]/metric[3])*100)

        return (epoch_met[0]/N_ADV_EPOCHS, epoch_met[1]/N_ADV_EPOCHS, epoch_met[2]/N_ADV_EPOCHS, history)


    # predict the adversarial network
    def predict_adv(self, data_iter):

        # Set the model to evaluation mode
        if isinstance(self.cl_model, torch.nn.Module):
            self.cl_model.eval()
            self.adv_model.eval()

        # collect all predictions, porbabilities and truths
        y_pred = []
        y_prob = []
        y_true = []
        s_prob = []

        metric = Accumulator(4)  # No. of correct predictions for classifier, adversary and no. of predictions
        with torch.no_grad():
            for X, y, s in data_iter:
                X, y, s = X.to(self.device), y.to(self.device), s.to(self.device)

                # predict probabilities for both models
                output = self.cl_model(X)

                ad_input = torch.cat((output, y), 1)
                p_z = self.adv_model(ad_input)

                # save adversary predictions
                p_z_temp = p_z.data.cpu().numpy()
                s_prob.extend(p_z_temp)

                # calculate metrics
                metric.add(self.accuracy(output, y), self.get_adv_accuracy(p_z, s), y.numel(), y.numel())

                # save binary predictions
                if self.non_binary[0]:
                    y_hat = (torch.argmax(output, dim=1)).data.cpu().numpy() #send to cpu to save
                    y_pred.extend(y_hat) # Save prediction
                else:
                    y_hat = (torch.round(output)).data.cpu().numpy() #send to cpu to save
                    y_pred.extend(y_hat) # Save prediction

                # calculate metrics
                metric.add(self.accuracy(output, y), self.get_adv_accuracy(p_z, s), y.numel())

                # save calssifier probabilities
                y_probs = output.data.cpu().numpy() #send to cpu to save
                y_prob.extend(y_probs)

                # save ground truth
                labels = y.data.cpu().numpy()
                y_true.extend(labels)


        # convert to array for storage and analysis
        ypred = np.asarray(y_pred)
        ytrue = np.asarray(y_true)
        yprob = np.asarray(y_prob)
        sprob = np.asarray(s_prob)

        print(f'classifier validation accuracy: {metric[0]/metric[2]*100}')
        print(f'adversary accuracy: {metric[1]/metric[2]*100}')

        # Return predictions and accuracy
        return  (ypred, yprob, ytrue, sprob, (metric[0]/metric[2])*100, (metric[1]/metric[2])*100)

    # predict the adversarial network
    def predict_adv_equal_opp(self, data_iter):

        # Set the model to evaluation mode
        if isinstance(self.cl_model, torch.nn.Module):
            self.cl_model.eval()  
            self.adv_model.eval()

        # collect all predictions, porbabilities and truths
        y_pred = []
        y_prob = []
        y_true = []
        s_prob = []

        metric = Accumulator(4)  # No. of correct predictions for classifier, adversary and no. of predictions
        with torch.no_grad():
            for X, y, s in data_iter:
                X, y, s = X.to(self.device), y.to(self.device), s.to(self.device)

                # predict probabilities for both models
                output = self.cl_model(X)

                # grab only when y = 1
                idx = self.get_idx(y)

                ad_input = torch.cat((output[idx], y[idx]), 1)
                p_z = self.adv_model(ad_input)

                # save adversary predictions
                p_z_temp = p_z.data.cpu().numpy()
                s_prob.extend(p_z_temp)

                # calculate metrics
                metric.add(self.accuracy(output, y), self.get_adv_accuracy(p_z, s[idx]), y.numel(), y[idx].numel())

                # save calssifier probabilities
                y_probs = output.data.cpu().numpy() #send to cpu to save
                y_prob.extend(y_probs)

                # save binary predictions
                y_hat = (torch.round(output)).data.cpu().numpy() #send to cpu to save
                y_pred.extend(y_hat) # Save prediction

                # save ground truth
                labels = y.data.cpu().numpy()
                y_true.extend(labels)


        # convert to array for storage and analysis
        ypred = np.asarray(y_pred)
        ytrue = np.asarray(y_true)
        yprob = np.asarray(y_prob)
        sprob = np.asarray(s_prob)

        print(f'classifier validation accuracy: {metric[0]/metric[2]*100}')
        print(f'adversary accuracy: {metric[1]/metric[3]*100}')

        # Return predictions and accuracy
        return  (ypred, yprob, ytrue, sprob, (metric[0]/metric[2])*100, (metric[1]/metric[3])*100)


    def adv_test_predict(self, data_iter):

        # Set the model to evaluation mode
        if isinstance(self.cl_model, torch.nn.Module):
            self.cl_model.eval()

        # collect all predictions, porbabilities and truths
        y_pred = []
        y_prob = []
        y_true = []

        metric = Accumulator(3)  # No. of correct predictions for classifier, adversary and no. of predictions
        with torch.no_grad():
            for X, y, s in data_iter:
                X, y, s = X.to(self.device), y.to(self.device), s.to(self.device)

                # predict probabilities for both models
                output = self.cl_model(X)

                # calculate metrics
                metric.add(self.accuracy(output, y), y.numel())

                # save classifier probabilities
                y_probs = output.data.cpu().numpy() #send to cpu to save
                y_prob.extend(y_probs)

                # save binary predictions
                y_hat = (torch.round(output)).data.cpu().numpy() #send to cpu to save
                y_pred.extend(y_hat) # Save prediction

                # save ground truth
                labels = y.data.cpu().numpy()
                y_true.extend(labels)


        # convert to array for storage and analysis
        ypred = np.asarray(y_pred)
        ytrue = np.asarray(y_true)
        yprob = np.asarray(y_prob)

        print(f'classifier validation accuracy: {metric[0]/metric[1]*100}')

        # Return predictions and accuracy
        return  (ypred, yprob, ytrue, (metric[0]/metric[1])*100)


    # collect the indices for when no zero values within a given tensor
    def get_idx(self, tensor):
        idx = torch.nonzero(tensor)
        return idx[:,0]


    # the projection, the norm of the current gradient parameter
    def normalize(self, x):
        return x / (torch.linalg.norm(x) + torch.finfo(torch.float32).tiny)# adding a small value to stop a divide by zero


        # evaluate the accuracy of the model
    def accuracyBCE(self, y_hat, y):
        y_hat = torch.round(y_hat)
        cmp = self.astype(y_hat, y.dtype) == y # convert y_hat into a data type that matches y
        correct = float(self.reduce_sum(self.astype(cmp, y.dtype))) # Taking the sum yields the number of correct predictions.
        return correct


    def accuracyCE(self, y_hat, y):
        y_hat = torch.argmax(y_hat)
        cmp = self.astype(y_hat, y.dtype) == y # convert y_hat into a data type that matches y
        correct = float(self.reduce_sum(self.astype(cmp, y.dtype))) # Taking the sum yields the number of correct predictions.
        return correct


    def get_adv_accuracyBCE(self, s_pred, s_true):
        s_p = torch.round(s_pred)
        cmp = self.astype(s_p, s_true.dtype) == s_true # convert y_hat into a data type that matches y 
        correct = float(self.reduce_sum(self.astype(cmp, s_true.dtype))) # Taking the sum yields the number of correct predictions.
        return correct


    def get_adv_accuracyCE(self, s_pred, s_true):
        s_p = torch.argmax(s_pred, dim=1)
        s_t = torch.argmax(s_true, dim=1)
        cmp = self.astype(s_p, s_t.dtype) == s_t # convert y_hat into a data type that matches y 
        correct = float(self.reduce_sum(self.astype(cmp, s_t.dtype))) # Taking the sum yields the number of correct predictions.
        return correct

    # grab the subet from the data given kfold idxs
    def get_kfold_data(self, train_idx, val_idx):
        train_sub = Subset(self.dataset, train_idx)
        val_sub = Subset(self.dataset, val_idx)
        return train_sub, val_sub

    def get_folds_data(self):
        return (self.all_folds_history, self.all_folds_predictions, self.all_folds_probabilities, self.all_folds_true, self.all_folds_val)

    def get_val_idxs(self):
        return self.all_val_idx

    def get_predictions(self):
        return self.all_folds_predictions

    def reset_data(self):
        self.all_folds_history = {}
        self.all_folds_predictions = {}
        self.all_folds_probabilities = {}
        self.all_folds_true = {}
        self.all_folds_val = {}
        self.all_val_idx = {}

    def set_model(self, state_dict):
        self.model.load_state_dict(state_dict)

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
