import sklearn
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

import logging
logger = logging.getLogger(__name__)

from ..featurizer import descriptors
from ..models.training import evaluate_performance

def get_baseline(mols, task_labels, mask_missing, cv, metrics=[]):
    logger.info(f'Start calculating baseline featues')

    features_num = np.hstack([descriptors.calc_RDKIT(mols)[0], descriptors.calc_MORGANF(mols)[0]])
    features_num[~np.isfinite(features_num)] = 0

    HYPERPARAMETERS = {
        "bootstrap": False,
        "criterion": "mse",
        "max_depth": 30,
        "max_features": 0.7,
        "max_leaf_nodes": None,
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "min_weight_fraction_leaf": 0.0,
        "n_estimators": 300,
        "oob_score": False,
        "verbose": 0,
        "warm_start": False
    }

    base_estimator = ExtraTreesRegressor()

    results = {}
    for cv_run, (train_idx, test_idx) in enumerate(cv):
        logger.info(f'Start Baseline CV run {cv_run + 1}/{len(cv)} with {len(train_idx)}/{len(test_idx)} compounds')

        preds = []
        std = []
        for task_id in range(task_labels.shape[1]):
            logger.info(f'Starting task {task_id}')

            y_train = task_labels[train_idx, task_id]
            desc_train = features_num[train_idx, :]
            missing_pat = mask_missing[train_idx, task_id].astype(int)

            y_train = y_train[missing_pat != 0]
            desc_train = desc_train[missing_pat != 0]

            estimator = sklearn.base.clone(base_estimator)
            estimator.set_params(**HYPERPARAMETERS)
            estimator.n_jobs = -1
            estimator.fit(desc_train, y_train)

            desc_test = features_num[test_idx, :]

            yhat = [e.predict(desc_test) for e in estimator.estimators_]
            preds.append(np.mean(yhat, axis=0))
            std.append(np.std(yhat, axis=0))

        preds = np.vstack(preds).transpose()
        std = np.vstack(std).transpose()

        metrics_results = {}
        if metrics:
            for metric in metrics:
                metrics_results[metric.__name__] = evaluate_performance(task_labels[test_idx], preds, mask_missing[test_idx], metric)

        results[cv_run] = {'test_metrics': metrics_results, 'test_preds': preds, 'test_std': std, 'test_y': task_labels[test_idx], 'test_idx': test_idx}


    return results
