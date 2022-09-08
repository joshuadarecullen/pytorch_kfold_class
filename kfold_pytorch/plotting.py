import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from aif360.metrics import ClassificationMetric
import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric


# plot single line
def plot_line(metric, lam, df, path=None, xlabel=None):
    stats = df.index[0:]
    if metric == 'val_acc' or metric == 'cl_val_acc':
        metric_idx = np.where(stats==metric)
        stat = np.array([item/100 for sublist in df.iloc[int(metric_idx[0])].values for item in sublist])
        metric = 'Classifier Accuracy'
    else:
        metric_idx = np.where(stats==metric)
        stat = get_row(metric,df,stats)

    plt.plot(lam,stat)
    plt.xlabel(xlabel, fontweight ='bold')
    plt.ylabel(metric, fontweight ='bold')

    plt.xticks(lam)
    plt.tight_layout()

    if path:
        plt.savefig(path)

    plt.show()


# plotting multplies lines graphs 
def plot_multi_line(metrics, lam, df, path=None, xlabel=None, ylabel=None):
    stats = df.index[0:]

    plt.rc('axes', labelsize=12) 
    plt.rc('xtick', labelsize=12) 
    plt.rc('ytick', labelsize=12) 
    plt.rc('legend', fontsize=12)

    for metric in metrics:

        if metric == 'val_acc' or metric == 'cl_val_acc':
            metric_idx = np.where(stats==metric)
            stat = np.array([item/100 for sublist in df.iloc[int(metric_idx[0])].values for item in sublist])
            metric = 'Classifier Accuracy'

        else:
            metric_idx = np.where(stats==metric)
            stat = get_row(metric,df,stats)

        plt.plot(lam,stat, label = metric)

    plt.xlabel(xlabel, fontweight ='bold')
    plt.ylabel(ylabel, fontweight ='bold')
    plt.xticks(lam)
    plt.legend()
    plt.tight_layout()

    if path:
        plt.savefig(path)

    plt.show()


def plot_adv(dep, metric, wd, indep, path=None):       
    legend = []
    for i,(k,v) in enumerate(wd.items()):
        plt.plot(indep,dep[i])
        legend.append(f'weight decay: {str(v)}')

    plt.xlabel('Adversarial Loss Weight', fontweight ='bold')
    plt.ylabel(metric, fontweight ='bold')
    plt.legend(legend, bbox_to_anchor=(1,1), loc="upper left")

    if path:
        plt.savefig(path)

    plt.show()


def plot_best(metrics, df,wd, path):
    stats = df.index[0:]
    selected_stats = []
    best_idx = []

    for metric in metrics:
        if metric == 'val_acc' or metric == 'cl_val_acc':
            metric_idx = np.where(stats==metric)
            accuracy = np.array([item/100 for sublist in df.iloc[int(metric_idx[0])].values for item in sublist])
            selected_stats.append(accuracy)
            idx_max = accuracy.argmax() 
            best_idx.append(idx_max)

        elif metric == 'bal_acc':
            stat = get_row(metric,df,stats)
            selected_stats.append(stat)
            idx_max = stat.argmax()
            best_idx.append(idx_max)

        else:
            stat = get_row(metric,df,stats)
            selected_stats.append(stat)
            idx = find_nearest(stat, 0 )
            best_idx.append(idx)

    best_scores = np.array(selected_stats) # create array of the selected stats
    final = np.array(best_scores[:,best_idx]) # indexing best metrics
    lams_best = [str(wd[x]) for x in best_idx] # get best lambdas
    y = [x + '/' + i for x,i in zip(metrics, lams_best)]

    plot_bar(X=final, Y=y, xlabel='best metrics/lambda', ylabel='metrics', bar_amount=len(metrics), legend=metrics, figsize=(8,8), path=path)


# a function to display the difference in privileged and unprivleged outcomes in the original data
def data_distr(dataset_orig, unprivileged_groups=None, privileged_groups=None, legend=[], labels=[], path=None):

    ds_metrics = BinaryLabelDatasetMetric(dataset_orig, unprivileged_groups=unprivileged_groups, 
                                      privileged_groups=privileged_groups)

    print(f'Statistical parity: {ds_metrics.statistical_parity_difference()}')
    print(f'Disparate impact: {ds_metrics.disparate_impact()}')

    # collect outcome stats
    stats = []

    # collecting the number of negative outcomes in the dataset
    num_negs = ds_metrics.num_negatives(privileged=None)
    priv_negs = ds_metrics.num_negatives(privileged=True)
    unpriv_negs = ds_metrics.num_negatives(privileged=None) - ds_metrics.num_negatives(privileged=True)
    negs = [num_negs, priv_negs, unpriv_negs]
    stats.append(negs)


    # collectin gnumber of positive outcomes in the dataset
    num_pos = ds_metrics.num_positives(privileged=None)
    priv_pos = ds_metrics.num_positives(privileged=True)
    unpriv_pos = ds_metrics.num_positives(privileged=None) - ds_metrics.num_positives(privileged=True)
    pos = [num_pos, priv_pos, unpriv_pos]
    stats.append(pos)

    data = np.asarray(stats)/ (num_pos+num_negs) # get the proporation in terms of the whole dataset

    #plotting, tranpose for barchart function
    plot_bar(data.T, Y=labels, ylabel='Proportion of Total Outcomes', legend=legend, bar_amount=data.shape[1], figsize=(8,8), path=path)

    return data


# for plotting one or multiply bars on a single plot
def plot_bar(X, Y=None, xlabel=None, ylabel=None, yticks=None, xticks=None, legend=None, xlim=None,
         ylim=None, title='',figsize=(12, 8), bar_amount=1, bar_width = 0.4, path=None):

    plt.rc('axes', labelsize=17) 
    plt.rc('xtick', labelsize=17) 
    plt.rc('ytick', labelsize=17) 
    plt.rc('legend', fontsize=12)

    if bar_amount != 1:

        barWidth = 0.1
        fig = plt.subplots(figsize=figsize)
        br = np.arange(X.shape[1])

        for i in range(X.shape[0]):
        # Set position of bar on X axis
            if i > 0:
                br = [x + barWidth for x in br]

            values = [item for item in X[i]]
            label = legend[i]
            plt.bar(br, values, width = barWidth, edgecolor ='grey', label = label)
            if yticks:
                plt.yticks(values)

        # Adding Xticks
        plt.xlabel(xlabel, fontweight ='bold')
        plt.ylabel(ylabel, fontweight ='bold')
        plt.xticks([r + barWidth for r in range(X.shape[1])], Y)
        plt.legend()

    else:
        fig = plt.figure(figsize =(10, 7))
        plt.bar(X,Y, width = bar_width)
        # plt.xticks([r for r+bar_width in range(len(X))], Y)
        plt.xlabel(xlabel, fontweight ='bold')
        plt.ylabel(ylabel, fontweight = 'bold')

        if yticks:
            plt.yticks(yticks)

    if path:
        plt.savefig(path)

    plt.show()


# confusion matrix from the aif360 library
def confusion_matrix(true, ypred,  privileged_groups, unprivileged_groups, groups=[], flag=0):

    confusion_m = {}
    val_pred = true.copy()                                                                                                      

    if flag == 0:
        val_pred.labels = ypred
    else:
        # depending on what dataset is being used
        preds = np.where( ypred == 0, 2, ypred)
        val_pred.labels = preds

    # Fetching fairness metrics for the fold
    fairness_metric = ClassificationMetric(true, val_pred,
                                       unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)

    confusion_m['All'] = fairness_metric.binary_confusion_matrix(privileged=None)

    if len(groups) != 0:
        confusion_m[groups[0]] = fairness_metric.binary_confusion_matrix(privileged=True)
        confusion_m[groups[1]] = fairness_metric.binary_confusion_matrix(privileged=False)

    return pd.DataFrame(confusion_m)


def roc_curve(probs, y_test, title):
    # calculate the fpr and tpr for all thresholds of the classification
    auc = metrics.roc_auc_score(y_test, probs)
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)

    # method I: plt
    plt.title(f'Receiver Operating Characteristic: {title}')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return fpr, tpr, threshold, auc, roc_auc


def get_hyperparam_data(lam, wd_all_probs, wd_all_true, wd_all_preds):
    all_true_temp = []
    all_preds_temp = []
    all_probs_temp = []
    for (k, prob),(f, true) in zip(wd_all_probs[lam].items(), wd_all_true[lam].items()):
        all_true_temp.append(true)
        all_probs_temp.append(prob)
    for k, v in wd_all_preds[lam].items():
        all_preds_temp.append(v)

    return (all_true_temp, all_preds_temp, all_probs_temp)


def get_fold_data(dic):
    data_list = []
    for k,v in dic.items():
        for k1,v1 in v.items():
            data_list.append(v1)


def get_row(metric, df, stats):
    metric_idx = np.where(stats==metric)
    stat = np.array([item for sublist in df.iloc[int(metric_idx[0])] for item in sublist])
    return stat


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
