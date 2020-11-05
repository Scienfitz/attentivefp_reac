import numpy as np
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn import model_selection

import logging
logger = logging.getLogger(__name__)

from ..models.dgl import collate_molgraphs, collate_molgraphs2, collate_molgraphs_Ext, EnsembleAttFP, EnsembleAttFP2, EnsembleAttFP_Ext
from ..utils import earlystop


def train_single_epoch(model, data_loader, loss_criterion, optimizer, device):
    losses = []
    model.train()
    for batch_id, batch_data in enumerate(data_loader):
        bg, labels, masks = batch_data
        bg.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        prediction = model(bg, bg.ndata['hv'], bg.edata['e'])

        # Handle qualified values.
        # Mask -1 = >, -2 = <
        # If prediction is fullfilling the qualified criteria set loss to 0 by setting mask to 0
        if torch.any(masks < 0):
            prediction[(prediction > labels) & (masks == -1)] = labels[(prediction > labels) & (masks == -1)]
            prediction[(prediction < labels) & (masks == -2)] = labels[(prediction < labels) & (masks == -2)]
            masks[masks < 0] = 1.0

        loss = (loss_criterion(prediction, labels) * masks).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    total_score = np.mean(losses)

    return total_score

def train_single_epoch2(model, data_loader, loss_criterion, optimizer, device):
    losses = []
    model.train()
    for batch_id, batch_data in enumerate(data_loader):
        bg1, bg2, bg3, labels, masks = batch_data
        bg1.to(device)
        bg2.to(device)
        bg3.to(device)
        labels = labels.to(device)
        masks  = masks.to(device)

        prediction = model(bg1, bg2, bg3,
                           bg1.ndata['hv'], bg2.ndata['hv'], bg3.ndata['hv'],
                           bg1.edata['e'],  bg2.edata['e'],  bg3.edata['e'])

        # Handle qualified values.
        # Mask -1 = >, -2 = <
        # If prediction is fullfilling the qualified criteria set loss to 0 by setting mask to 0
        if torch.any(masks < 0):
            prediction[(prediction > labels) & (masks == -1)] = labels[(prediction > labels) & (masks == -1)]
            prediction[(prediction < labels) & (masks == -2)] = labels[(prediction < labels) & (masks == -2)]
            masks[masks < 0] = 1.0

        loss = (loss_criterion(prediction, labels) * masks).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    total_score = np.mean(losses)

    return total_score

def train_single_epoch_Ext(model, data_loader, loss_criterion, optimizer, device):
    losses = []
    model.train()
    for batch_id, batch_data in enumerate(data_loader):
        labels, masks, tabs, *graphs = batch_data

        g_data = []
        for g in graphs:
            g.to(device)
            g_data += [g, g.ndata['hv'], g.edata['e']]
        labels = labels.to(device)
        masks  = masks.to(device)
        tabs = tabs.to(device)

        prediction = model(tabs, *g_data)

        # Handle qualified values.
        # Mask -1 = >, -2 = <
        # If prediction is fullfilling the qualified criteria set loss to 0 by setting mask to 0
        if torch.any(masks < 0):
            prediction[(prediction > labels) & (masks == -1)] = labels[(prediction > labels) & (masks == -1)]
            prediction[(prediction < labels) & (masks == -2)] = labels[(prediction < labels) & (masks == -2)]
            masks[masks < 0] = 1.0

        loss = (loss_criterion(prediction, labels) * masks).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())

    total_score = np.mean(losses)

    return total_score


def eval_single_epoch(dataloader, model, loss_fn, device):
    val_losses = []
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            bg, labels, masks = batch_data
            bg.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            prediction = model(bg, bg.ndata['hv'], bg.edata['e'])

            # Handle qualified values.
            # Mask -1 = >, -2 = <
            # If prediction is fullfilling the qualified criteria set loss to 0 by setting mask to 0
            if torch.any(masks < 0):
                prediction[(prediction > labels) & (masks == -1)] = labels[(prediction > labels) & (masks == -1)]
                prediction[(prediction < labels) & (masks == -2)] = labels[(prediction < labels) & (masks == -2)]
                masks[masks < 0] = 1.0

            val_loss = (loss_fn(prediction, labels) * masks).sum() / masks.sum()
            val_losses.append(val_loss.data.item())

        val_score = np.mean(val_losses)

    return val_score

def eval_single_epoch2(dataloader, model, loss_fn, device):
    val_losses = []
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            bg1, bg2, bg3, labels, masks = batch_data
            bg1.to(device)
            bg2.to(device)
            bg3.to(device)
            labels = labels.to(device)
            masks  = masks.to(device)

            prediction = model(bg1, bg2, bg3,
                               bg1.ndata['hv'], bg2.ndata['hv'], bg3.ndata['hv'],
                               bg1.edata['e'],  bg2.edata['e'],  bg3.edata['e'])

            # Handle qualified values.
            # Mask -1 = >, -2 = <
            # If prediction is fullfilling the qualified criteria set loss to 0 by setting mask to 0
            if torch.any(masks < 0):
                prediction[(prediction > labels) & (masks == -1)] = labels[(prediction > labels) & (masks == -1)]
                prediction[(prediction < labels) & (masks == -2)] = labels[(prediction < labels) & (masks == -2)]
                masks[masks < 0] = 1.0

            val_loss = (loss_fn(prediction, labels) * masks).sum() / masks.sum()
            val_losses.append(val_loss.data.item())

        val_score = np.mean(val_losses)

    return val_score

def eval_single_epoch_Ext(dataloader, model, loss_fn, device):
    val_losses = []
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(dataloader):
            labels, masks, tabs, *graphs = batch_data

            g_data = []
            for g in graphs:
                g.to(device)
                g_data += [g, g.ndata['hv'], g.edata['e']]
            labels = labels.to(device)
            masks = masks.to(device)
            tabs = tabs.to(device)

            prediction = model(tabs, *g_data)

            # Handle qualified values.
            # Mask -1 = >, -2 = <
            # If prediction is fullfilling the qualified criteria set loss to 0 by setting mask to 0
            if torch.any(masks < 0):
                prediction[(prediction > labels) & (masks == -1)] = labels[(prediction > labels) & (masks == -1)]
                prediction[(prediction < labels) & (masks == -2)] = labels[(prediction < labels) & (masks == -2)]
                masks[masks < 0] = 1.0

            val_loss = (loss_fn(prediction, labels) * masks).sum() / masks.sum()
            val_losses.append(val_loss.data.item())

        val_score = np.mean(val_losses)

    return val_score


def training_dataloader(model, optimizer, train_loader, loss_fn, max_epochs=1000, bootstrap_runs=1, bootstrap_seed=None, patience=50, device=None):
    # bootstrap sampling. random 20% as validation set
    sub_data_splits = [(train_idx, test_idx) for train_idx, test_idx in
                       model_selection.ShuffleSplit(n_splits=bootstrap_runs, test_size=0.2, random_state=bootstrap_seed).split(train_loader.dataset)]

    # save original inputs
    init_model = model
    init_optimizer = optimizer
    init_pretrained = not all([p.requires_grad for p in init_model.parameters()])
    if init_pretrained:
        logger.info('Using pre-trained model as starting point')

    summary = []
    for model_id in range(bootstrap_runs):
        logger.info('Start bootstrap training %d/%d' % (model_id + 1, bootstrap_runs))

        # get the subset train/test for this run based on the bootstrap Shuffle splits
        train_idx, test_idx = sub_data_splits[model_id]
        sub_train_ds = [torch.utils.data.Subset(train_loader.dataset, train_idx), torch.utils.data.Subset(train_loader.dataset, test_idx)]
        # build data loaders from Subsets
        sub_train_dl = DataLoader(dataset=sub_train_ds[0], batch_size=train_loader.batch_size, collate_fn=collate_molgraphs)
        sub_val_dl = DataLoader(dataset=sub_train_ds[1], batch_size=train_loader.batch_size, collate_fn=collate_molgraphs)

        # get fresh copies
        model = copy.deepcopy(init_model)
        optimizer = torch.optim.Adam(model.parameters(), **init_optimizer.defaults)

        # set model to train mode
        model.to(device)
        model.train()

        train_scores = []
        val_scores = []

        stopper = earlystop.EarlyStopping(mode='lower', patience=1 if init_pretrained else patience)
        unfrozen = 0
        for e in range(max_epochs):
            train_score = train_single_epoch(model, sub_train_dl, loss_fn, optimizer, device)
            val_score = eval_single_epoch(sub_val_dl, model, loss_fn, device)

            train_scores.append(train_score)
            val_scores.append(val_score)

            if (e+1)%10: # log every 100 epochs to info
                logger.debug('epoch {:d}/{:d}, training {:.4f}, validation {:.4f}'.format(e+1, max_epochs, train_score, val_score))
            else:
                logger.info('epoch {:d}/{:d}, training {:.4f}, validation {:.4f}'.format(e+1, max_epochs, train_score, val_score))

            if patience > 0:
                early_stop = stopper.step(val_score, model)
                if early_stop:
                    # If a pretrained model was used, unfreeze parameters after first early stop reached and continue training
                    if init_pretrained and not unfrozen:
                        unfrozen = e
                        logger.info('Unfreezing parameters')
                        stopper.load_checkpoint(model) # will only load the state_dict into existing model

                        for param in model.parameters():
                            param.requires_grad = True

                        stopper = earlystop.EarlyStopping(mode='lower', patience=patience) # reset stopper
                    else:
                        logger.info('Early Stopping at epoch %d using checkpoint %d' % (e+1, stopper.best_step+unfrozen))
                        break

        if patience > 0:
            # get best model from early stopping checkpoint
            stopper.load_checkpoint(model)

        summary.append(
            {'model': model.cpu(), 'final_checkpoint': stopper.best_step+unfrozen, 'final_loss_train': train_scores[stopper.best_step+unfrozen], 'final_loss_val': val_scores[stopper.best_step+unfrozen], 'train_losses': train_scores, 'val_losses': val_scores})

    return summary

def training_dataloader2(model, optimizer, train_loader, loss_fn, max_epochs=1000, bootstrap_runs=1, bootstrap_seed=None, patience=50, device=None):
    # bootstrap sampling. random 20% as validation set
    sub_data_splits = [(train_idx, test_idx) for train_idx, test_idx in
                       model_selection.ShuffleSplit(n_splits=bootstrap_runs, test_size=0.2, random_state=bootstrap_seed).split(train_loader.dataset)]

    # save original inputs
    init_model      = model
    init_optimizer  = optimizer
    init_pretrained = not all([p.requires_grad for p in init_model.parameters()])
    if init_pretrained:
        logger.info('Using pre-trained model as starting point')

    summary = []
    for model_id in range(bootstrap_runs):
        logger.info('Start bootstrap training %d/%d' % (model_id + 1, bootstrap_runs))

        # get the subset train/test for this run based on the bootstrap Shuffle splits
        train_idx, test_idx = sub_data_splits[model_id]
        sub_train_ds        = [torch.utils.data.Subset(train_loader.dataset, train_idx), torch.utils.data.Subset(train_loader.dataset, test_idx)]
        # build data loaders from Subsets
        sub_train_dl = DataLoader(dataset=sub_train_ds[0], batch_size=train_loader.batch_size, collate_fn=collate_molgraphs2)
        sub_val_dl   = DataLoader(dataset=sub_train_ds[1], batch_size=train_loader.batch_size, collate_fn=collate_molgraphs2)

        # get fresh copies
        model     = copy.deepcopy(init_model)
        optimizer = torch.optim.Adam(model.parameters(), **init_optimizer.defaults)

        # set model to train mode
        model.to(device)
        model.train()

        train_scores = []
        val_scores = []

        stopper = earlystop.EarlyStopping(mode='lower', patience=1 if init_pretrained else patience)
        unfrozen = 0
        for e in range(max_epochs):
            train_score = train_single_epoch2(model, sub_train_dl, loss_fn, optimizer, device)
            val_score   = eval_single_epoch2(sub_val_dl, model, loss_fn, device)

            train_scores.append(train_score)
            val_scores.append(val_score)

            if (e+1)%10: # log every 100 epochs to info
                logger.debug('epoch {:d}/{:d}, training {:.4f}, validation {:.4f}'.format(e+1, max_epochs, train_score, val_score))
            else:
                logger.info('epoch {:d}/{:d}, training {:.4f}, validation {:.4f}'.format(e+1, max_epochs, train_score, val_score))

            if patience > 0:
                early_stop = stopper.step(val_score, model)
                if early_stop:
                    # If a pretrained model was used, unfreeze parameters after first early stop reached and continue training
                    if init_pretrained and not unfrozen:
                        unfrozen = e
                        logger.info('Unfreezing parameters')
                        stopper.load_checkpoint(model) # will only load the state_dict into existing model

                        for param in model.parameters():
                            param.requires_grad = True

                        stopper = earlystop.EarlyStopping(mode='lower', patience=patience) # reset stopper
                    else:
                        logger.info('Early Stopping at epoch %d using checkpoint %d' % (e+1, stopper.best_step+unfrozen))
                        break

        if patience > 0:
            # get best model from early stopping checkpoint
            stopper.load_checkpoint(model)

        summary.append(
            {'model': model.cpu(), 'final_checkpoint': stopper.best_step+unfrozen, 'final_loss_train': train_scores[stopper.best_step+unfrozen], 'final_loss_val': val_scores[stopper.best_step+unfrozen], 'train_losses': train_scores, 'val_losses': val_scores})

    return summary

def training_dataloader_Ext(model, optimizer, train_loader, loss_fn, max_epochs=1000, bootstrap_runs=1, bootstrap_seed=None, patience=50, device=None):
    # bootstrap sampling. random 20% as validation set
    sub_data_splits = [(train_idx, test_idx) for train_idx, test_idx in
                       model_selection.ShuffleSplit(n_splits=bootstrap_runs, test_size=0.2, random_state=bootstrap_seed).split(train_loader.dataset)]

    # save original inputs
    init_model      = model
    init_optimizer  = optimizer
    init_pretrained = not all([p.requires_grad for p in init_model.parameters()])
    if init_pretrained:
        logger.info('Using pre-trained model as starting point')

    summary = []
    for model_id in range(bootstrap_runs):
        logger.info('Start bootstrap training %d/%d' % (model_id + 1, bootstrap_runs))

        # get the subset train/test for this run based on the bootstrap Shuffle splits
        train_idx, test_idx = sub_data_splits[model_id]
        sub_train_ds        = [torch.utils.data.Subset(train_loader.dataset, train_idx), torch.utils.data.Subset(train_loader.dataset, test_idx)]
        # build data loaders from Subsets
        sub_train_dl = DataLoader(dataset=sub_train_ds[0], batch_size=train_loader.batch_size, collate_fn=collate_molgraphs_Ext)
        sub_val_dl   = DataLoader(dataset=sub_train_ds[1], batch_size=train_loader.batch_size, collate_fn=collate_molgraphs_Ext)

        # get fresh copies
        model     = copy.deepcopy(init_model)
        optimizer = torch.optim.Adam(model.parameters(), **init_optimizer.defaults)

        # set model to train mode
        model.to(device)
        model.train()

        train_scores = []
        val_scores = []

        stopper = earlystop.EarlyStopping(mode='lower', patience=1 if init_pretrained else patience)
        unfrozen = 0
        for e in range(max_epochs):
            train_score = train_single_epoch_Ext(model, sub_train_dl, loss_fn, optimizer, device)
            val_score   = eval_single_epoch_Ext(sub_val_dl, model, loss_fn, device)

            train_scores.append(train_score)
            val_scores.append(val_score)

            if (e+1)%10: # log every 100 epochs to info
                logger.debug('epoch {:d}/{:d}, training {:.4f}, validation {:.4f}'.format(e+1, max_epochs, train_score, val_score))
            else:
                logger.info('epoch {:d}/{:d}, training {:.4f}, validation {:.4f}'.format(e+1, max_epochs, train_score, val_score))

            if patience > 0:
                early_stop = stopper.step(val_score, model)
                if early_stop:
                    # If a pretrained model was used, unfreeze parameters after first early stop reached and continue training
                    if init_pretrained and not unfrozen:
                        unfrozen = e
                        logger.info('Unfreezing parameters')
                        stopper.load_checkpoint(model) # will only load the state_dict into existing model

                        for param in model.parameters():
                            param.requires_grad = True

                        stopper = earlystop.EarlyStopping(mode='lower', patience=patience) # reset stopper
                    else:
                        logger.info('Early Stopping at epoch %d using checkpoint %d' % (e+1, stopper.best_step+unfrozen))
                        break

        if patience > 0:
            # get best model from early stopping checkpoint
            stopper.load_checkpoint(model)

        summary.append(
            {'model': model.cpu(), 'final_checkpoint': stopper.best_step+unfrozen, 'final_loss_train': train_scores[stopper.best_step+unfrozen], 'final_loss_val': val_scores[stopper.best_step+unfrozen], 'train_losses': train_scores, 'val_losses': val_scores})

    return summary


def predict_dataloader(dataloader, model, device=None, dropout=False):
    model = model.to(device)

    if dropout:
        model.train()
    else:
        model.eval()

    all_pred = []
    with torch.no_grad():
        for batch in dataloader:
            bg, _, _ = batch
            bg.to(device)
            pred = model(bg, bg.ndata['hv'], bg.edata['e'])
            all_pred.append(pred.data.cpu())

    return torch.cat(all_pred, dim=-2)

def predict_dataloader2(dataloader, model, device=None, dropout=False):
    model = model.to(device)

    if dropout:
        model.train()
    else:
        model.eval()

    all_pred = []
    with torch.no_grad():
        for batch in dataloader:
            bg1, bg2, bg3, _, _ = batch
            bg1.to(device)
            bg2.to(device)
            bg3.to(device)
            pred = model(bg1, bg2, bg3,
                         bg1.ndata['hv'], bg2.ndata['hv'], bg3.ndata['hv'],
                         bg1.edata['e'],  bg2.edata['e'],  bg3.edata['e'])
            all_pred.append(pred.data.cpu())

    return torch.cat(all_pred, dim=-2)

def predict_dataloader_Ext(dataloader, model, device=None, dropout=False):
    model = model.to(device)

    if dropout:
        model.train()
    else:
        model.eval()

    all_pred = []
    with torch.no_grad():
        for batch in dataloader:
            _, _, tabs, *graphs = batch

            g_data = []
            for g in graphs:
                g.to(device)
                g_data += [g, g.ndata['hv'], g.edata['e']]

            if tabs is not None:
                tabs = tabs.to(device)

            pred = model(tabs, *g_data)
            all_pred.append(pred.data.cpu())

    return torch.cat(all_pred, dim=-2)


def predict(graphs, model, device, batch_size=1000, dropout_samples=0):
    dataloader = DataLoader(
        list(zip(graphs, torch.zeros(len(graphs)))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_molgraphs
    )

    if dropout_samples:
        preds = torch.stack([predict_dataloader(dataloader, model, device=device, dropout=True) for _ in range(dropout_samples)], dim=0)
    else:
        preds = predict_dataloader(dataloader, model, device, dropout=False)

    if len(preds.shape) > 3:
        preds = preds.view(np.prod(preds.shape[:-2]), preds.shape[-2], preds.shape[-1])

    if len(preds.shape) > 2:
        y_hat = preds.mean(0)
        std = preds.std(0)
    else:
        y_hat = preds
        std = torch.zeros(y_hat.shape)

    return y_hat, std

def predict2(g1, g2, g3, model, device, batch_size=1000, dropout_samples=0):
    dataloader = DataLoader(
        list(zip(g1, g2, g3, torch.zeros(len(g1)))),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_molgraphs2
    )

    if dropout_samples:
        preds = torch.stack([predict_dataloader2(dataloader, model, device=device, dropout=True) for _ in range(dropout_samples)], dim=0)
    else:
        preds = predict_dataloader2(dataloader, model, device, dropout=False)

    if len(preds.shape) > 3:
        preds = preds.view(np.prod(preds.shape[:-2]), preds.shape[-2], preds.shape[-1])

    if len(preds.shape) > 2:
        y_hat = preds.mean(0)
        std   = preds.std(0)
    else:
        y_hat = preds
        std   = torch.zeros(y_hat.shape)

    return y_hat, std

def predict_Ext(model, device, tabs, *graphs, batch_size=1000, dropout_samples=0):
    dataloader = DataLoader(
        list(zip(torch.zeros(len(graphs[0])), torch.zeros(len(graphs[0])), tabs, *graphs)),
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_molgraphs_Ext
    )

    if dropout_samples:
        preds = torch.stack([predict_dataloader_Ext(dataloader, model, device=device, dropout=True) for _ in range(dropout_samples)], dim=0)
    else:
        preds = predict_dataloader_Ext(dataloader, model, device, dropout=False)

    if len(preds.shape) > 3:
        preds = preds.view(np.prod(preds.shape[:-2]), preds.shape[-2], preds.shape[-1])

    if len(preds.shape) > 2:
        y_hat = preds.mean(0)
        std   = preds.std(0)
    else:
        y_hat = preds
        std   = torch.zeros(y_hat.shape)

    return y_hat, std


def evaluate_performance(labels, predictions, masks, metric):
    metric_scores = []
    for d in range(labels.shape[1]):
        mask = masks[:,d] != 0
        y    = labels[mask,d]
        pred = predictions[mask, d]

        if not len(y):
            logger.info(f'No test data for task {d}')
            metric_scores.append(np.nan)
            continue

        try:
            metric_scores.append(metric(y, pred))
        except:
            logger.exception(f'metric failed to calculate for task {d}')
            metric_scores.append(np.nan)

    return metric_scores


def perform_cv(model, optimizer, graphs, task_labels, mask_missing, cv, loss_fn, metrics=[], max_epochs=1000, bootstrap_runs=1, bootstrap_seed=None, batch_size=128, patience=50, device=None):
    results = {}
    for cv_run, (train_idx, test_idx) in enumerate(cv):
        logger.info(f'Start CV run {cv_run + 1}/{len(cv)} with {len(train_idx)}/{len(test_idx)} compounds')

        # Get train subset for training and run training
        train_dataset = list(zip(graphs[train_idx], task_labels[train_idx], mask_missing[train_idx]))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_molgraphs
        )

        run_summary = training_dataloader(model, optimizer, train_dataloader, loss_fn=loss_fn, patience=patience, device=device, bootstrap_runs=bootstrap_runs, bootstrap_seed=bootstrap_seed, max_epochs=max_epochs)

        # use Bootstrap model ensemble to make predictions
        # setup ensemble
        ensemble = EnsembleAttFP(models=[r['model'] for r in run_summary])
        # score
        preds, std = predict(graphs[test_idx], ensemble, device=device, batch_size=1000, dropout_samples=0)

        metrics_results = {}
        if metrics:
            for metric in metrics:
                metrics_results[metric.__name__] = evaluate_performance(task_labels[test_idx].numpy(), preds.numpy(), mask_missing[test_idx].numpy(), metric)

        results[cv_run] = {'test_metrics': metrics_results, 'test_preds': preds, 'test_std': std, 'test_y': task_labels[test_idx].numpy(), 'test_idx': test_idx}

    return results

def perform_cv2(model, optimizer, g1, g2, g3, task_labels, mask_missing, cv, loss_fn, metrics=[], max_epochs=1000, bootstrap_runs=1, bootstrap_seed=None, batch_size=128, patience=50, device=None):
    results = {}
    for cv_run, (train_idx, test_idx) in enumerate(cv):
        logger.info(f'Start CV run {cv_run + 1}/{len(cv)} with {len(train_idx)}/{len(test_idx)} compounds')

        # Get train subset for training and run training
        train_dataset = list(zip(g1[train_idx], g2[train_idx], g3[train_idx], task_labels[train_idx], mask_missing[train_idx]))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = 0,
            collate_fn  = collate_molgraphs2
        )

        run_summary = training_dataloader2(model, optimizer, train_dataloader, loss_fn=loss_fn, patience=patience, device=device, bootstrap_runs=bootstrap_runs, bootstrap_seed=bootstrap_seed, max_epochs=max_epochs)

        # use Bootstrap model ensemble to make predictions
        # setup ensemble
        ensemble = EnsembleAttFP2(models=[r['model'] for r in run_summary])
        # score
        preds, std = predict2(g1[test_idx], g2[test_idx], g3[test_idx], ensemble, device=device, batch_size=1000, dropout_samples=0)

        metrics_results = {}
        if metrics:
            for metric in metrics:
                metrics_results[metric.__name__] = evaluate_performance(task_labels[test_idx].numpy(), preds.numpy(), mask_missing[test_idx].numpy(), metric)

        results[cv_run] = {'test_metrics': metrics_results, 'test_preds': preds, 'test_std': std, 'test_y': task_labels[test_idx].numpy(), 'test_idx': test_idx}

    return results

def perform_cv_Ext(model, optimizer, task_labels, mask_missing, tabs, cv, loss_fn, *graphs, metrics=[], max_epochs=1000, bootstrap_runs=1, bootstrap_seed=None, batch_size=128, patience=50, device=None):
    results = {}
    for cv_run, (train_idx, test_idx) in enumerate(cv):
        logger.info(f'Start CV run {cv_run + 1}/{len(cv)} with {len(train_idx)}/{len(test_idx)} compounds')

        # Get train subset for training and run training
        train_graphs = [g[train_idx] for g in graphs]
        test_graphs  = [g[test_idx] for g in graphs]
        train_dataset = list(zip(task_labels[train_idx], mask_missing[train_idx], tabs[train_idx], *train_graphs))

        train_dataloader = DataLoader(
            train_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = 0,
            collate_fn  = collate_molgraphs_Ext
        )

        run_summary = training_dataloader_Ext(model, optimizer, train_dataloader, loss_fn=loss_fn, patience=patience, device=device, bootstrap_runs=bootstrap_runs, bootstrap_seed=bootstrap_seed, max_epochs=max_epochs)

        # use Bootstrap model ensemble to make predictions
        # setup ensemble
        ensemble = EnsembleAttFP_Ext(models=[r['model'] for r in run_summary])
        # score
        preds, std = predict_Ext(ensemble, device, tabs[test_idx], *test_graphs, batch_size=1000, dropout_samples=0)

        metrics_results = {}
        if metrics:
            for metric in metrics:
                metrics_results[metric.__name__] = evaluate_performance(task_labels[test_idx].numpy(), preds.numpy(), mask_missing[test_idx].numpy(), metric)

        results[cv_run] = {'test_metrics': metrics_results, 'test_preds': preds, 'test_std': std, 'test_y': task_labels[test_idx].numpy(), 'test_idx': test_idx}

    return results