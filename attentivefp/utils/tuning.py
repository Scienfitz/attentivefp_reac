import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, space_eval, pyll

import logging
logger = logging.getLogger(__name__)

from ..models.dgl import AttentiveFPDense2, collate_molgraphs2, AttentiveFPDense, collate_molgraphs, AttentiveFPDense_Ext, collate_molgraphs_Ext
from ..models.training import training_dataloader, training_dataloader2, training_dataloader_Ext


def hyperopt(graphs, task_labels, mask_missing, hyperparams, max_evals, max_epochs, patience, device, seed):
    logger.info(f'Running {max_evals} hyperparameter optimization trials')

    # if no seed, fix a seed for evals
    if seed is None:
        seed = np.random.randint(1000)

    space = {
        'node_feat_size':  hyperparams['node_feat_size'],
        'edge_feat_size':  hyperparams['edge_feat_size'],
        'num_layers':      hp.quniform('num_layers', 1, 4, 1),
        'num_timesteps':   hp.quniform('num_timesteps', 1, 4, 1),
        'graph_feat_size': hp.quniform('graph_feat_size', 150, 400, 10),
        'dropout':         hp.quniform('dropout', 0, 0.5, 0.05),
        'n_units':         hp.quniform('n_units', 100, 500, 25),
        'n_dense':         hp.quniform('n_dense', 0, 3, 1),
        'lr':              hp.quniform('lr', -4.5, -2.0, 0.05),
        'weight_decay':    0,
        'batch_size':      512
    }

    local_dataloader = DataLoader(
        list(zip(graphs, task_labels, mask_missing)),
        batch_size=128,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_molgraphs
    )

    def hp_objective(hyp):
        local_model = AttentiveFPDense(node_feat_size= hyperparams['node_feat_size'],
                                 edge_feat_size      = hyperparams['edge_feat_size'],
                                 num_layers          = int(hyp['num_layers']),
                                 num_timesteps       = int(hyp['num_timesteps']),
                                 graph_feat_size     = int(hyp['graph_feat_size']),
                                 dropout             = hyp['dropout'],
                                 n_dense             = int(hyp['n_dense']),
                                 n_units             = int(hyp['n_units']),
                                 n_tasks             = task_labels.shape[1]
                                 )
        local_model = local_model.to(device)
        local_optimizer = torch.optim.Adam(local_model.parameters(),
                                     lr=np.power(10, hyp['lr']),
                                     weight_decay=hyp['weight_decay'])

        # run a single training using a fixed seed for comparison
        summary = training_dataloader(local_model, local_optimizer, local_dataloader,
                                      loss_fn=nn.SmoothL1Loss(reduction='none'),
                                      patience=patience, device=device, bootstrap_runs=1,
                                      bootstrap_seed=seed, max_epochs=max_epochs)

        return summary[0]['final_loss_val']

    trials = Trials()
    best = fmin(hp_objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_loss = trials.average_best_error()
    best_params = space_eval(space, best)
    print(best_loss)
    print(best_params)

    if hyperparams is not None:
        base_loss = hp_objective(hyperparams)
        print(base_loss)
        logger.info(f'Validation loss baseline: {base_loss}, hypopt: {best_loss}')
        if base_loss <= best_loss:
            logger.info('Baseline parameters better than optimized hyperparameters. Using baseline')
            best_params = hyperparams
    else:
        logger.info(f'Best loss found: {best_loss}')

    return best_params

def hyperopt2(graphs1, graphs2, graphs3, task_labels, mask_missing, hyperparams, max_evals, max_epochs, patience, device, seed, batch_size=512):
    # Augmented Version: graphs is now a list of lists. entries of each list will be treated as a separate graph
    logger.info(f'Running {max_evals} hyperparameter optimization trials')

    # if no seed, fix a seed for evals
    if seed is None:
        seed = np.random.randint(1000)

    space = {
        'node_feat_size':  hyperparams['node_feat_size'],
        'edge_feat_size':  hyperparams['edge_feat_size'],
        'num_layers':      hp.quniform('num_layers', 1, 4, 1),
        'num_timesteps':   hp.quniform('num_timesteps', 1, 4, 1),
        'graph_feat_size': hp.quniform('graph_feat_size', 150, 400, 10),
        'dropout':         hp.quniform('dropout', 0, 0.5, 0.05),
        'n_units':         hp.quniform('n_units', 100, 600, 25),
        'n_dense':         hp.quniform('n_dense', 0, 3, 1),
        'lr':              hp.quniform('lr', -4.0, -2.0, 0.05),
        'weight_decay':    0,
        'batch_size':      batch_size
    }

    local_dataloader = DataLoader(
        list(zip(graphs1, graphs2, graphs3, task_labels, mask_missing)),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = collate_molgraphs2
    )

    def hp_objective(hyp):
        local_model = AttentiveFPDense2(node_feat_size  = hyperparams['node_feat_size'],
                                        edge_feat_size  = hyperparams['edge_feat_size'],
                                        num_layers      = int(hyp['num_layers']),
                                        num_timesteps   = int(hyp['num_timesteps']),
                                        graph_feat_size = int(hyp['graph_feat_size']),
                                        dropout         = hyp['dropout'],
                                        n_dense         = int(hyp['n_dense']),
                                        n_units         = int(hyp['n_units']),
                                        n_tasks         = task_labels.shape[1],
                                        )
        local_model     = local_model.to(device)
        local_optimizer = torch.optim.Adam(local_model.parameters(),
                                           lr           = np.power(10, hyp['lr']),
                                           weight_decay = hyp['weight_decay'])

        # run a single training using a fixed seed for comparison
        summary = training_dataloader2(local_model, local_optimizer, local_dataloader,
                                       loss_fn = nn.SmoothL1Loss(reduction='none'), patience = patience, device=device,
                                       bootstrap_runs=1, bootstrap_seed = seed, max_epochs=max_epochs)

        return summary[0]['final_loss_val']

    trials = Trials()
    best = fmin(hp_objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_loss   = trials.average_best_error()
    best_params = space_eval(space, best)
    print(best_loss)
    print(best_params)

    if hyperparams is not None:
        base_loss = hp_objective(hyperparams)
        print(base_loss)
        logger.info(f'Validation loss baseline: {base_loss}, hypopt: {best_loss}')
        if base_loss <= best_loss:
            logger.info('Baseline parameters better than optimized hyperparameters. Using baseline')
            best_params = hyperparams
    else:
        logger.info(f'Best loss found: {best_loss}')

    return best_params

def hyperopt_Ext(task_labels, mask_missing, tabs, hyperparams, device, *graphs, max_evals=20, max_epochs=1000, patience=100, seed=None, batch_size=512, nHPRounds=3):
    # Augmented Version: graphs is now a list of lists. entries of each list will be treated as a separate graph
    logger.info(f'Running {max_evals} hyperparameter optimization trials')

    # if no seed, fix a seed for evals
    if seed is None:
        seed = np.random.randint(1337)

    space = {
        'node_feat_size':  hyperparams['node_feat_size'],
        'edge_feat_size':  hyperparams['edge_feat_size'],
        'n_graphs':        hyperparams['n_graphs'],
        'tab_feat_size':   hyperparams['tab_feat_size'],
        'num_layers':      hp.quniform('num_layers', 1, 4, 1),
        'num_timesteps':   hp.quniform('num_timesteps', 1, 4, 1),
        'graph_feat_size': hp.quniform('graph_feat_size', 150, 400, 10),
        'dropout':         hp.quniform('dropout', 0, 0.5, 0.05),
        'n_units':         hp.quniform('n_units', 100, 600, 25),
        'n_dense':         hp.quniform('n_dense', 0, 3, 1),
        'lr':              hp.quniform('lr', -4.0, -2.0, 0.05),
        'weight_decay':    0,
        'batch_size':      batch_size,
    }

    local_dataloader = DataLoader(
        list(zip(task_labels, mask_missing, tabs, *graphs)),
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = collate_molgraphs_Ext
    )

    def hp_objective(hyp):
        res = []
        for kRun in range(nHPRounds):
            print(f'\n\n##################### HP Round {kRun+1}\n')
            local_model = AttentiveFPDense_Ext(node_feat_size  = hyperparams['node_feat_size'],
                                               edge_feat_size  = hyperparams['edge_feat_size'],
                                               n_graphs        = hyperparams['n_graphs'],
                                               tab_feat_size   = hyperparams['tab_feat_size'],
                                               num_layers      = int(hyp['num_layers']),
                                               num_timesteps   = int(hyp['num_timesteps']),
                                               graph_feat_size = int(hyp['graph_feat_size']),
                                               dropout         = hyp['dropout'],
                                               n_dense         = int(hyp['n_dense']),
                                               n_units         = int(hyp['n_units']),
                                               n_tasks         = task_labels.shape[1])
            local_model     = local_model.to(device)
            local_optimizer = torch.optim.Adam(local_model.parameters(),
                                               lr           = np.power(10, hyp['lr']),
                                               weight_decay = hyp['weight_decay'])

            # run a single training using a fixed seed for comparison
            summary = training_dataloader_Ext(local_model, local_optimizer, local_dataloader,
                                           loss_fn = nn.SmoothL1Loss(reduction='none'), patience = patience, device=device,
                                           bootstrap_runs=1, bootstrap_seed = seed+kRun, max_epochs=max_epochs)
            res.append(summary[0]['final_loss_val'])
        return np.mean(res)

    trials = Trials()
    best = fmin(hp_objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_loss   = trials.average_best_error()
    best_params = space_eval(space, best)
    print(best_loss)
    print(best_params)

    if hyperparams is not None:
        base_loss = hp_objective(hyperparams)
        print(base_loss)
        logger.info(f'Validation loss baseline: {base_loss}, hypopt: {best_loss}')
        if base_loss <= best_loss:
            logger.info('Baseline parameters better than optimized hyperparameters. Using baseline')
            best_params = hyperparams
    else:
        logger.info(f'Best loss found: {best_loss}')

    return best_params
