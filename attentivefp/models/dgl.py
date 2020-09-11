import dgl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
from dgllife.model.model_zoo.attentivefp_predictor import AttentiveFPPredictor


class AttentiveFPDense(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers=3,
                 num_timesteps=2,
                 graph_feat_size=200,
                 dropout=0.2,
                 n_dense=0,
                 n_units=256,
                 n_tasks=1):
        super(AttentiveFPDense, self).__init__()

        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.graph_feat_size = graph_feat_size
        self.dropout = dropout
        self.n_dense = n_dense
        self.n_units = n_units
        self.n_tasks = n_tasks

        self.attfp = AttentiveFPPredictor(node_feat_size=node_feat_size,
                                          edge_feat_size=edge_feat_size,
                                          num_layers=num_layers,
                                          num_timesteps=num_timesteps,
                                          graph_feat_size=graph_feat_size,
                                          dropout=dropout,
                                          n_tasks=n_tasks)


        if n_dense > 0:
            # disable dgllife attfp predict layer by replacing with nn.Identity
            self.attfp.predict = nn.Identity()
            self.dense = []
            for d in range(n_dense):
                self.dense.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(n_units if d > 0 else graph_feat_size, n_units),
                        nn.ReLU()
                    )
                )
            self.dense = nn.ModuleList(self.dense)
            self.predict = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_units, n_tasks))


    def summary_dict(self):
        return {'node_feat_size': self.node_feat_size,
                'edge_feat_size': self.edge_feat_size,
                'num_layers': self.num_layers,
                'num_timesteps': self.num_timesteps,
                'graph_feat_size': self.graph_feat_size,
                'dropout': self.dropout,
                'n_units': self.n_units,
                'n_dense': self.n_dense,
                'n_tasks': self.n_tasks
                }

    def forward(self, g, node_feats, edge_feats):
        x = self.attfp(g, node_feats, edge_feats)

        if self.n_dense > 0:
            for i in range(len(self.dense)):
                x = self.dense[i](x)
            x = self.predict(x)

        return x


class EnsembleAttFP(nn.Module):
    def __init__(self, models):
        super(EnsembleAttFP, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, g, node_feats, edge_feats):
        output = torch.stack([model(g, node_feats, edge_feats) for model in self.models], dim=0)
        return output


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [2, 3], \
        'Expect the tuple to be of length 2 or 3, got {:d}'.format(len(data[0]))
    if len(data[0]) == 2:
        graphs, labels = map(list, zip(*data))
        masks = None
    else:
        graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)

    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    if labels is None:
        labels = torch.ones(labels.shape)
    else:
        labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return bg, labels, masks


class AttentiveFPDense_tab(nn.Module):
    def __init__(self,
                 node_feat_size,
                 edge_feat_size,
                 num_layers      = 3,
                 num_timesteps   = 2,
                 graph_feat_size = 200,
                 tab_feat_size   = 100,
                 dropout         = 0.2,
                 n_dense         = 0,
                 n_units         = 256,
                 n_tasks         = 1):
        super(AttentiveFPDense_tab, self).__init__()

        self.node_feat_size  = node_feat_size
        self.edge_feat_size  = edge_feat_size
        self.num_layers      = num_layers
        self.num_timesteps   = num_timesteps
        self.graph_feat_size = graph_feat_size
        self.tab_feat_size   = tab_feat_size
        self.dropout = dropout
        self.n_dense = n_dense
        self.n_units = n_units
        self.n_tasks = n_tasks

        self.attfp = AttentiveFPPredictor(node_feat_size=node_feat_size,
                                          edge_feat_size=edge_feat_size,
                                          num_layers=num_layers,
                                          num_timesteps=num_timesteps,
                                          graph_feat_size=graph_feat_size,
                                          dropout=dropout,
                                          n_tasks=n_tasks)


        if n_dense > 0:
            # disable dgllife attfp predict layer by replacing with nn.Identity
            self.attfp.predict = nn.Identity()
            self.dense = []
            for d in range(n_dense):
                self.dense.append(
                    nn.Sequential(
                        nn.Dropout(dropout),
                        nn.Linear(n_units if d > 0 else graph_feat_size + tab_feat_size, n_units),
                        nn.ReLU()
                    )
                )
            self.dense = nn.ModuleList(self.dense)
            self.predict = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_units, n_tasks))


    def summary_dict(self):
        return {'node_feat_size':  self.node_feat_size,
                'edge_feat_size':  self.edge_feat_size,
                'num_layers':      self.num_layers,
                'num_timesteps':   self.num_timesteps,
                'graph_feat_size': self.graph_feat_size,
                'tab_feat_size':   self.tab_feat_size,
                'dropout': self.dropout,
                'n_units': self.n_units,
                'n_dense': self.n_dense,
                'n_tasks': self.n_tasks
                }

    def forward(self, g, node_feats, edge_feats, tab_feats):
        x1 = self.attfp(g, node_feats, edge_feats)
        x2 = tab_feats
        x = torch.cat([x1,x2], dim=1)

        if self.n_dense > 0:
            for i in range(len(self.dense)):
                x = self.dense[i](x)
            x = self.predict(x)

        return x


class EnsembleAttFP_tab(nn.Module):
    def __init__(self, models):
        super(EnsembleAttFP_tab, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, g, node_feats, edge_feats, tab_feats):
        output = torch.stack([model(g, node_feats, edge_feats, tab_feats) for model in self.models], dim=0)
        return output


def collate_molgraphs_tab(data):
    """Batching a list of datapoints for dataloader. Includes tabular features.
    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.
    Returns
    -------
    bg : BatchedDGLGraph
        Batched DGLGraphs
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        graphs, labels, tabs = map(list, zip(*data))
        masks = None
    else:
        graphs, labels, tabs, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)

    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    if labels is None:
        labels = torch.ones(labels.shape)
    else:
        labels = torch.stack(labels, dim=0)

    if tabs is None:
        tabs = torch.zeros(tabs.shape)
    else:
        tabs = torch.stack(tabs, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return bg, labels, tabs, masks