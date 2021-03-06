import os
import sys
from glob import glob
import argparse
import pandas as pd
import torch
import numpy as np
import json

import logging
logger = logging.getLogger(__name__)

from rdkit import Chem
from attentivefp.featurizer.graph import DGLFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from attentivefp.featurizer.otherfeaturizers import ReactionAtomFeaturizer, ReactionAttentiveFPAtomFeaturizer
from attentivefp.utils import log
from attentivefp.models.training import predict as att_predict
from attentivefp.utils.chem import standardize_mol, smiles2mol, mol2smiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description='AttentiveFP Predictions')
    parser.add_argument('-d', '--dataset', help='Input file', required=True)
    parser.add_argument('-m', '--model_dir', help='Model directory', required=True)
    parser.add_argument('-o', '--outfile', help='Output file', required=False, default=None)
    parser.add_argument('-s', '--smiles_col', help='SMILES column', default=None, required=False)
    parser.add_argument('-v', '--verbose', help='Verbosity', type=int, default=0)
    parser.add_argument('-idx', '--index_col', help='Index columns', required=False, default=None)
    parser.add_argument('-test', '--test', help='Test on small data subset', action='store_true')
    parser.add_argument('-drp', '--dropout', help='Number of samples for dropout Bayesian uncertainty approximation', type=int, default=0)
    parser.add_argument('-std', '--standardize', help='Standardize input molecules', action='store_true')
    parser.add_argument('--featurizer', help='Featurizer Type (Default: Canonical)', action='store', required=False, default="Canonical", choices=['Canonical','Reaction','ReactionFP','AttentiveFP'])
    args = parser.parse_args()
    return args

def predict(data_file, model_dir, smiles_col=None, index_col=None, test=False, dropout_samples=0, standardize=False):
    """Calculate predictions of user-specified SMILES.

    Arguments:
        dataset {pandas} -- pandas dataset with smiles column
        model {str} -- Location of model .pth file

    Keyword Arguments:
        smiles_col {str} -- SMILES column name. Leave as None if file has no column names. (default: {None})
        test {bool} -- Run test mode (default: {False})

    Returns:
        Prediction results and standard deviation as tensors
    """

    # get model
    logging.info('load model')

    _model_json_file = os.path.join(model_dir, 'model.json')
    if not os.path.isfile(_model_json_file):
        logging.error(f'No model definition model.json found in {model_dir}')
        raise FileNotFoundError(f'No model definition model.json found in {model_dir}')

    with open(_model_json_file, 'r') as f:
        _model_json = json.load(f)

    _model_pth_file = os.path.join(model_dir, 'model.pth')
    if not os.path.isfile(_model_json_file):
        logging.error(f'No model.pth file found in {model_dir}')
        raise FileNotFoundError(f'No model.pth file found in {model_dir}')

    columns = _model_json['columns']
    model =  torch.load(_model_pth_file, map_location=device).to(device)

    # chunk data to avoid large amounts of data in memory
    header = None if smiles_col is None else 'infer'
    for chunk_id, chunk in enumerate(pd.read_csv(data_file, sep=None, engine='python', chunksize=10000, header=header, nrows=100 if test else None)):
        # load data
        if smiles_col is not None:
            if smiles_col not in chunk.columns:
                logging.error(f'column {smiles_col} does not exist in input')
                raise ValueError(f'column {smiles_col} does not exist in input')
        else:
            if len(chunk.columns ) > 1:
                logging.warning(f'Detected {len(chunk.columns)} columns without smiles column defined. Will use column 0 as smiles')
            smiles_col = chunk.columns[0]

        if index_col is not None:
            if index_col not in chunk.columns:
                logging.error(f'index {index_col} does not exist in input')
            else:
                chunk.index = chunk[index_col]

        logger.info(f'Start processing chunk {chunk_id} with {len(chunk)} rows')

        # Get mols & standardize
        #
        chunk['_mol'] = chunk[smiles_col].apply(smiles2mol)
        if standardize:
            chunk['_mol'] = chunk['_mol'].apply(standardize_mol)
            # put canonical smiles into column
            chunk['_smiles'] = chunk['_mol'].apply(mol2smiles)

        missing_mols = np.where(chunk['_mol'].isna())[0]
        if len(missing_mols) > 0:
            missing_smiles = chunk[smiles_col].iloc[missing_mols].values.astype(str)
            logger.info(f'Smiles failed to convert into mol: {",".join(missing_smiles)}')

        # convert mols into DGL graphs
        logging.debug(f'calculate graphs')
        
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
    
        dglf = DGLFeaturizer(device=device, AtomFeaturizer=atom_featurizer, BondFeaturizer=bond_featurizer)
        graphs = dglf.featurize_mols(chunk['_mol'])

        good_idx = np.where(np.not_equal(graphs, None))[0]
        graphs = graphs[good_idx]

        logging.debug(f'start prediction for {len(graphs)} compounds')
        preds, std = att_predict(graphs, model, device, batch_size=2000, dropout_samples=dropout_samples)

        att_df = pd.DataFrame(np.concatenate([preds, std], axis=1), index=good_idx, columns=columns + [f'{c}:std' for c in columns])
        att_df = att_df.reindex(range(len(chunk)), axis=0)

        # set original index
        att_df.index = chunk.index

        # add std smiles & error message
        if standardize:
            att_df['_smiles'] = chunk['_smiles']
        if len(missing_mols) > 0:
            att_df.loc[att_df.index[missing_mols],'error'] = 'Smiles conversion failed'

        logging.info(f'Finished chunk {chunk_id}')
        yield att_df

if __name__ == '__main__':
    args = parse_arguments()

    log.initialize_logger(args.verbose)

    _input_file_path = args.dataset if os.path.isabs(args.dataset) else os.path.join(os.getcwd(), args.dataset)
    if not os.path.isfile(_input_file_path):
        logging.error(f'input file not found: {_input_file_path}')
        raise IOError(f'input file not found: {_input_file_path}')

    _model_file_path = args.model_dir if os.path.isabs(args.model_dir) else os.path.join(os.getcwd(), args.model_dir)
    if not os.path.isdir(_model_file_path):
        logging.error(f'model directory not found: {_model_file_path}')
        raise IOError(f'model directory not found: {_model_file_path}')

    if args.outfile is None:
        _output_file_writer = sys.stdout
    else:
        _output_file_path = args.outfile if os.path.isabs(args.outfile) else os.path.join(os.getcwd(), args.outfile)
        _output_file_writer = open(_output_file_path, 'w')

    # verify smiles_col exist in input data
    smiles_col = args.smiles_col
    df = pd.read_csv(_input_file_path, sep=None, engine='python', nrows=5)
    if smiles_col and  smiles_col not in df.columns:
        logging.error(f'column {smiles_col} does not exist in input  {_input_file_path}')
        raise ValueError(f'column {smiles_col} does not exist in input {_input_file_path}')

    index_col = args.index_col
    if index_col and index_col not in df.columns:
        logging.error(f'index {index_col} does not exist in input  {_input_file_path}')
        index_col = None
    del df

    # perform predictions and output results
    try:
        for cid, df in enumerate(predict(data_file=_input_file_path, model_dir=_model_file_path, smiles_col=smiles_col, index_col=index_col, test=args.test, dropout_samples=args.dropout, standardize=args.standardize)):
            df.reindex(sorted(df.columns), axis=1).to_csv(_output_file_writer, sep='\t', index=True, index_label="ID", float_format='%.2f', header=False if cid > 0 else True)
            _output_file_writer.flush()
    except Exception as e:
        logger.exception(str(e))

    # Close properly
    logging.shutdown()
    sys.stdin.close()
    sys.stdout.close()
