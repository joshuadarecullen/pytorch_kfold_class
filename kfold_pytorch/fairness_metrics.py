from aif360.metrics import ClassificationMetric
from utils import Accumulator
import numpy as np

# For use in getting fairness given the indices of each fold
def get_fairness_metrics(k, true_data, predictions, unprivileged_groups, privileged_groups, val_idxs, flag=0):

    fair_metrics = Accumulator(5)
    fairness_history = {}

    fold_history = {'stat_par_diff': [], 'eq_opp_diff': [],
                'avg_odds_diff': [], 'bal_acc': [], 'disp_imp': []}

    i = 1
    for (key,pred), (key1,val_idx) in zip(predictions.items(), val_idxs.items()):

        # setting the models preditictions to corresponding features in original aif data
        true_data = true_data.copy()
        val_true = true_data.subset(val_idx)
        val_pred = val_true.copy()

        # depending on what dataset is being used
        if flag == 0:
            val_pred.labels = pred
        else:
            preds = np.where(pred == 0, 2, pred)
            val_pred.labels = preds

        # Fetching fairness metrics for the fold
        fairness_metric = ClassificationMetric(val_true, val_pred,
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

        stat_par = fairness_metric.statistical_parity_difference()
        equal_opp = fairness_metric.equal_opportunity_difference()
        aver_odds = fairness_metric.average_odds_difference()
        true_pos = (fairness_metric.true_positive_rate() + fairness_metric.true_negative_rate()) / 2
        dis_par = (fairness_metric.disparate_impact())


        # Accumulate for computing averages
        fair_metrics.add(stat_par, equal_opp, aver_odds, true_pos, dis_par)

        # Keep track of fairness metrics
        fold_history['stat_par_diff'].append(stat_par)
        fold_history['eq_opp_diff'].append(equal_opp)
        fold_history['avg_odds_diff'].append(aver_odds)
        fold_history['bal_acc'].append(true_pos)
        fold_history['disp_imp'].append(dis_par)

        fairness_history[f'fold{i}'] = fold_history

        i += 1

    # Compute and save averages
    kfolds_avergs = { 'stat_par_diff': [fair_metrics[0]/k],
                     'eq_opp_diff': [fair_metrics[1]/k],
                     'avg_odds_diff': [fair_metrics[2]/k],
                     'bal_acc': [fair_metrics[3]/k],
                     'disp_imp': [fair_metrics[4]/k]}

    return kfolds_avergs, fold_history
