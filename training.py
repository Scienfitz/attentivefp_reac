import os
import time
import argparse
import pandas as pd
import numpy as np
import pickle
import random
import json
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import dgl
import dgllife
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from attentivefp.featurizer.otherfeaturizers import ReactionAtomFeaturizer, ReactionAttentiveFPAtomFeaturizer
from dgllife.utils import one_hot_encoding
from dgllife.utils import EarlyStopping
from dgllife.utils import mol_to_bigraph
from dgllife.model import model_zoo

from functools import partial
from sklearn.metrics import roc_auc_score, r2_score, median_absolute_error, mean_squared_error
from sklearn import model_selection

from rdkit import Chem
from rdkit import RDPaths
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

import logging
logger = logging.getLogger(__name__)

from collections import defaultdict

from attentivefp.utils import chem, log, data, splitter, tuning
from attentivefp.featurizer import graph
from attentivefp.models.dgl import AttentiveFPDense, collate_molgraphs, AttentiveFPDense2, collate_molgraphs2, EnsembleAttFP
from attentivefp.models import baseline
from attentivefp.models.training import perform_cv, training_dataloader, perform_cv2, training_dataloader2

# define appropriate torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MISSING_VALUE_FILL = -999

def parse_arguments():
    parser = argparse.ArgumentParser(description='AttentiveFP Training')
    parser.add_argument('-t', '--task_cols',  help='Tasks column names', nargs='+', required=True)
    parser.add_argument('-d', '--dataset',    help='Input file', required=True)
    #parser.add_argument('-s', '--smiles_col', help='SMILES column name', required=True)
    parser.add_argument('-s', '--smiles_cols', help='SMILES column names', nargs='+', required=True)
    parser.add_argument('-id', '--id_col',    help='ID column name', required=False)
    parser.add_argument('-o', '--outdir',     help='Output directory', required=True)
    parser.add_argument('-v', '--verbose',    help='Verbosity', type=int, default=0)
    parser.add_argument('--split',            help='CrossValidation Split Type (Default: KFold)', action='store', required=False, default="KFold", choices=['KFold','Random','Butina','Scaffold','GroupKFold','Predefined','Cumulative'])
    parser.add_argument('--split_column',     help='Column with splitting data. Only relevant for split types "GroupKFold" and "Predefined"', action='store', required=False, dest='split_column')
    parser.add_argument('--split_n',          help='Number of splits to perform', type=int, default=5)
    parser.add_argument('--bootstrap_n',      help='Number of bootstrap models to build', type=int, default=5)
    parser.add_argument('--hyper_evals',      help='Hyperopt max_evals. 0 disabled hyperoptimization and uses defaults', type=int, default=0)
    parser.add_argument('--max_epochs',       help='Maximum number of training epochs', type=int, default=1000)
    parser.add_argument('--patience',         help='Training EarlyStopping patience', type=int, default=100)
    parser.add_argument('--test',             help='Test on small data subset', action='store_true')
    parser.add_argument('--standardize',      help='Standardize input molecules', action='store_true')
    parser.add_argument('--baseline',         help='Evaluate baseline model on same split', action='store_true')
    parser.add_argument('--skip_final',       help='Do NOT train a final model', action='store_true')
    parser.add_argument('--skip_cv',          help='Do NOT perform any cross validation', action='store_true')
    parser.add_argument('--pretrained_model', help='pth model file location of trained AttentiveFPDense model', required=False)
    parser.add_argument('--seed',             help='Seed for data randomization, cross validation, boot strapping, and torch', required=False, type=int, default=None)
    #### Added arguments
    parser.add_argument('--featurizer',       help='Featurizer Type (Default: Canonical)', action='store', required=False, default="Canonical", choices=['Canonical','Reaction','ReactionFP','AttentiveFP'])
    parser.add_argument('--frac',             help='Only do calculations on a fraction of all data, e.g. 0.2', required=False, type=float, default=1.0)
    parser.add_argument('--separate_graphs',  help='Instead of one disconnected graph use 3 separate graphs', action='store_true')
    return parser.parse_args()

def main(args):
    input_file = args.dataset if os.path.isabs(args.dataset) else os.path.join(os.getcwd(), args.dataset)
    if not os.path.isfile(input_file):
        logging.error(f'input file not found: {input_file}')
        raise IOError(f'input file not found: {input_file}')

    save_dir = args.outdir if os.path.isabs(args.outdir) else os.path.join(os.getcwd(), args.outdir)
    save_dir = save_dir.rstrip('/')
    if os.path.isdir(save_dir):
        logger.warning(f'Output directory {save_dir} already exists')
    else:
        os.makedirs(save_dir)

    #if args.seed is not None:
    #    torch.manual_seed(args.seed)

    df = pd.read_csv(input_file, sep=None, engine='python', nrows=100 if args.test else None)
    logger.info(f'Imported {df.shape[0]} rows with columns: {", ".join(df.columns)}')

    if args.frac != 1.0:
        df = df.sample(frac=args.frac, random_state=args.seed)
        logger.info(f'Reduced dataframe to {df.shape[0]} rows with columns: {", ".join(df.columns)}')

    task_cols   = args.task_cols
    smiles_cols = args.smiles_cols
    id_col      = args.id_col
    grp_col     = args.split_column
    logger.info(f'Using task columns {",".join(task_cols)} for training')

    # preprocess data
    # will result in reduced df containing only valid rows
    # Define mask for missing values and get torch representations
    df_prep, task_labels, mask_missing = data.preprocess(df, smiles_cols, task_cols , id_col, grp_col, args.standardize, random_state=args.seed)
    del df

    task_labels[mask_missing == 0] = MISSING_VALUE_FILL # ensure a finite value for all labels
    lst_mols = [df_prep[f'_mol{k}'].values for k,_ in enumerate(smiles_cols)]
    #mols = df_prep['_mol'].values
    grp  = df_prep['_grp'].values if grp_col else None

    if not args.skip_cv or args.hyper_evals or args.baseline:
        # Note that we only pass the first series of mols to the splitter.
        # Splittings that consider info from all mol series are not supported yet
        cvFolds = splitter.getSplit(lst_mols[0], task_labels, split=args.split, n_splits=args.split_n, groups=grp, random_state=args.seed)

        # put fold info in input df
        df_prep['CV_fold'] = -1 # -1 indicates to be used always in train set
        for fold, (_, test_ids) in enumerate(cvFolds):
            df_prep.loc[df_prep.index[test_ids], 'CV_fold'] = fold
        df_prep['CV_fold'] = df_prep['CV_fold'].astype(int)

    # save processed input data to file
    #df_prep.drop('_mol', axis=1).sort_index().to_csv(os.path.join(save_dir,'processed_dataset.txt'), sep='\t', index_label='IDX') #disable saving this for now

    # Set Featurizers
    if args.featurizer=='AttentiveFP':
        atom_featurizer = AttentiveFPAtomFeaturizer(atom_data_field='hv')
        bond_featurizer = AttentiveFPBondFeaturizer()
    elif args.featurizer=='Reaction':
        atom_featurizer = ReactionAtomFeaturizer(atom_data_field='hv')
        bond_featurizer = CanonicalBondFeaturizer()
    elif args.featurizer=='ReactionFP':
        atom_featurizer = ReactionAttentiveFPAtomFeaturizer(atom_data_field='hv')
        bond_featurizer = AttentiveFPBondFeaturizer()
    else: #Use Canonical otherwise
        atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='hv')
        bond_featurizer = CanonicalBondFeaturizer()
    
    atomFeatureSize=len(atom_featurizer(Chem.MolFromSmiles('CC'))['hv'][0])
    bondFeatureSize=len(bond_featurizer(Chem.MolFromSmiles('CC'))['e'][0])
    
    # Generate graphs for DGL
    dglf = graph.DGLFeaturizer(device=device, AtomFeaturizer=atom_featurizer, BondFeaturizer=bond_featurizer)
    #graphs = dglf.featurize_mols(mols)
    lst_graphs = [dglf.featurize_mols(mols) for mols in lst_mols]
    print(lst_graphs)

    hyperparameters = {'node_feat_size': atomFeatureSize,#74 for canonical
                       'edge_feat_size': bondFeatureSize,#12 for canonical
                       'num_layers':      3,
                       'num_timesteps':   2,
                       'graph_feat_size': 200,
                       'dropout':        0.2,
                       'n_units':        128,
                       'n_dense':        0,
                       'lr':             -3.5,
                       'weight_decay':   0,
                       'batch_size':     256
                       }

    if args.hyper_evals:
        # do hyper evals
        hyperparameters = tuning.hyperopt2(lst_graphs, task_labels, mask_missing, hyperparameters, args.hyper_evals, args.max_epochs, args.patience, device, args.seed)

    hyperparameters['n_tasks'] = task_labels.shape[1]

    if args.pretrained_model is not None:
        model = torch.load(args.pretrained_model)
        # # Freeze AttFP model weights
        # for param in model.attfp.parameters():
        #     param.requires_grad = False
        # Freeze AttFP model weights
        for attfp in model.attfps:
            for param in attfp.parameters():
                param.requires_grad = False
        # replace final layer with new layer with correct number of tasks
        model.predict = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                      nn.Linear(in_features=model.predict[1].in_features, out_features=hyperparameters['n_tasks'], bias=True))
    else:
        model = AttentiveFPDense2(node_feat_size=atomFeatureSize,
                                  edge_feat_size=bondFeatureSize,
                                  num_layers=int(hyperparameters['num_layers']),
                                  num_timesteps=int(hyperparameters['num_timesteps']),
                                  graph_feat_size=int(hyperparameters['graph_feat_size']),
                                  dropout=int(hyperparameters['dropout']),
                                  n_dense=int(hyperparameters['n_dense']),
                                  n_units=int(hyperparameters['n_units']),
                                  n_tasks=int(hyperparameters['n_tasks']),
                                  n_graphs=len(lst_graphs)
                                  )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=np.power(10,hyperparameters['lr']),
                                 weight_decay=hyperparameters['weight_decay'])

    if args.split != "None" and not args.skip_cv:
        train_dl = perform_cv2(model, optimizer, lst_graphs, task_labels, mask_missing, cvFolds,
                               loss_fn=nn.SmoothL1Loss(reduction='none'), max_epochs=args.max_epochs,
                               bootstrap_seed=None, bootstrap_runs=args.bootstrap_n,
                               patience=args.patience, device=device, batch_size=int(hyperparameters['batch_size']),
                               metrics=[mean_squared_error, r2_score, median_absolute_error])
        summary_metrics = defaultdict(list)

        with open(os.path.join(save_dir, 'cv_results.txt'), 'w') as _output_file_writer:
            for k, d in train_dl.items():
                for m, x in d['test_metrics'].items():
                    summary_metrics[m].append(x)

                # replace missing value dummy value with NA for export
                d['test_y'][d['test_y'] == MISSING_VALUE_FILL] = np.nan
                # Export CV results to file
                cv_df = pd.DataFrame(np.concatenate([d['test_y'], d['test_preds'], d['test_std']], axis=1), columns=[f'{c}' for c in task_cols] + [f'{c}:pred' for c in task_cols] + [f'{c}:std' for c in task_cols])
                cv_df = cv_df.reindex(sorted(cv_df.columns), axis=1)
                cv_df.insert(loc=0, column='IDX', value=df_prep.index[d['test_idx']].values)
                cv_df.insert(loc=1, column='ID', value=df_prep.iloc[d['test_idx']]['_id'].values)
                cv_df.insert(loc=2, column='Fold', value=k+1)
                cv_df.to_csv(_output_file_writer, sep='\t', index=False, float_format='%.3f', header=False if k > 0 else True)

        logger.info('CV metrics:')
        with open(os.path.join(save_dir, 'cv_summary.txt'), 'w') as _output_file_writer:
            csv_writer = csv.writer(_output_file_writer, delimiter='\t')
            csv_writer.writerow(['Metric','Fold'] + [f'{c}' for c in task_cols])
            for m, d in summary_metrics.items():
                logger.info(f'    {m}, {np.asarray(d).mean(axis=0)}')
                for k, sub in enumerate(d):
                    csv_writer.writerow([m, k+1] + [f'{x:.3f}' for x in sub])

    # Disabled for now
    # if args.baseline:
    #     bsl_r = baseline.get_baseline(mols, task_labels.numpy(), mask_missing.numpy(), cvFolds, metrics=[mean_squared_error, r2_score, median_absolute_error])
    #     baseline_summary_metrics = defaultdict(list)
    #
    #     with open(os.path.join(save_dir, 'baseline_results.txt'), 'w') as _output_file_writer:
    #         for k, d in bsl_r.items():
    #             for m, x in d['test_metrics'].items():
    #                 baseline_summary_metrics[m].append(x)
    #             # replace missing value dummy value with NA for export
    #             d['test_y'][d['test_y'] == MISSING_VALUE_FILL] = np.nan
    #             # Export CV results to file
    #             cv_df = pd.DataFrame(np.concatenate([d['test_y'], d['test_preds'], d['test_std']], axis=1), columns=[f'{c}' for c in task_cols] + [f'{c}:pred' for c in task_cols] + [f'{c}:std' for c in task_cols])
    #             cv_df = cv_df.reindex(sorted(cv_df.columns), axis=1)
    #             cv_df.insert(loc=0, column='IDX', value=df_prep.index[d['test_idx']].values)
    #             cv_df.insert(loc=1, column='ID', value=df_prep.iloc[d['test_idx']]['_id'].values)
    #             cv_df.insert(loc=2, column='Fold', value=k+1)
    #             cv_df.to_csv(_output_file_writer, sep='\t', index=False, float_format='%.3f', header=False if k > 0 else True)
    #
    #     logger.info('Baseline metrics:')
    #     with open(os.path.join(save_dir, 'baseline_summary.txt'), 'w') as _output_file_writer:
    #         csv_writer = csv.writer(_output_file_writer, delimiter='\t')
    #         csv_writer.writerow(['Metric', 'Fold'] + [f'{c}' for c in task_cols])
    #         for m, d in baseline_summary_metrics.items():
    #             logger.info(f'    {m}, {np.asarray(d).mean(axis=0)}')
    #             for k, sub in enumerate(d):
    #                 csv_writer.writerow([m, k+1] + [f'{x:.3f}' for x in sub])

    # final model training using all data for export and deployment
    if not args.skip_final:
        logger.info(f'Training final model on all {len(lst_graphs[0])} datapoints')
        # Create data loader
        final_dataloader = DataLoader(
            list(zip(map(list,*lst_graphs), task_labels, mask_missing)),
            batch_size  = int(hyperparameters['batch_size']),
            shuffle     = True,
            num_workers = 0,
            collate_fn  = collate_molgraphs2
        )

        # run bootstrap training on all data
        summary = training_dataloader2(model, optimizer, final_dataloader, loss_fn=nn.SmoothL1Loss(reduction='none'),
                                       patience=args.patience, device=device, bootstrap_runs=args.bootstrap_n,
                                       bootstrap_seed=args.seed, max_epochs=args.max_epochs)

        ensemble_models = []
        for mid, r in enumerate(summary):
            m = r.pop('model')
            os.makedirs(os.path.join(save_dir, str(mid)), exist_ok=True)
            torch.save(
                m,
                os.path.join(save_dir, str(mid) ,f'model.pth'))
            ensemble_models.append(m)

        ensemble = EnsembleAttFP(models=ensemble_models)
        torch.save(
            ensemble,
            os.path.join(save_dir, f'model.pth'))
        torch.save(
            ensemble.state_dict(),
            os.path.join(save_dir, f'model.state_dict.pth'))
        pickle.dump(
            summary,
            open(os.path.join(save_dir, f'model.summary.pkl'), 'wb'))

        json.dump({'model_class':  model.__class__.__name__,
                   'model_module': model.__class__.__module__,
                   'columns':      list(task_cols),
                   'n_bootstrap':  len(summary),
                   'parameters':   hyperparameters,
                   },
                  open(os.path.join(save_dir, 'model.json'), 'w'))

        logger.info(f'Exported model file to {os.path.join(save_dir, "model.pth")}')

if __name__ == '__main__':
    args = parse_arguments()
    log.initialize_logger(args.verbose)

    logger.info(f"Using device: {device}")

    args.task_cols = np.hstack([x.split(',') for x in args.task_cols])

    # quick first data check
    _input_file_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(os.getcwd(), args.dataset)
    if not os.path.isfile(_input_file_path):
        logging.error(f'input file not found: {_input_file_path}')
        raise IOError(f'input file not found: {_input_file_path}')

    df = pd.read_csv(_input_file_path, sep=None, engine='python', nrows=5)
    assert all([l in df.columns for l in args.task_cols]), 'Required column missing in input'
    assert all([l in df.columns for l in args.smiles_cols]), f'smiles column {args.smiles_col} missing in input'
    if args.id_col:
        assert args.id_col in df.columns, f'ID column {args.id_col} missing in input'

    if args.pretrained_model is not None:
        assert args.hyper_evals == 0, 'Can\'t perform hyper parameter opt with pretrained model'

    try:
        main(args)
    except Exception as e:
        logger.exception(str(e))
