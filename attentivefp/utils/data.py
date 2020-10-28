import numpy as np
import torch
import pandas as pd
from ..utils.chem import smiles2mol, standardize_mol, mol2smiles, isEmptyMolecule
import logging
logger = logging.getLogger(__name__)


def preprocess(df, smiles_col, task_cols, id_col=None, grp_col=None, standardize=True, random_state=None):
    # verify input
    assert all([l in df.columns for l in task_cols]), 'Required column missing in input'
    assert smiles_col in df.columns, f'smiles column {smiles_col} missing in input'
    if id_col:
        assert id_col in df.columns, f'ID column {id_col} missing in input'
        df['_id'] = df[id_col]
    else:
        df['_id'] = df.index

    if grp_col:
        assert grp_col in df.columns, f'Split column {grp_col} missing in input'
        df['_grp'] = df[grp_col]

    # init randomization
    if random_state is not False:
        df = df.sample(frac=1, random_state=random_state)

    # Get mols & standardize
    #
    logger.info(f'Smiles 2 Mol')
    df['_mol'] = df[smiles_col].apply(smiles2mol)

    if standardize:
        logger.info('Run standardize')
        df['_mol'] = df['_mol'].apply(standardize_mol)

    # put canonical smiles into column
    df['_smiles'] = df['_mol'].apply(mol2smiles)

    # convert string columns
    mask_greater = df[task_cols].apply(lambda x: x.astype(str).str.contains('>'))
    mask_smaller = df[task_cols].apply(lambda x: x.astype(str).str.contains('<'))
    df.loc[:, df[task_cols].select_dtypes(exclude='number').columns] = df[task_cols].select_dtypes(exclude='number').apply(lambda x: pd.to_numeric(x.str.replace('<|>|=|~|\*', ''), errors='coerce'))

    good_ids = df.index[np.any(np.isfinite(df[task_cols]), axis=1)]
    good_ids = good_ids.intersection(df.index[~df['_mol'].apply(isEmptyMolecule)])

    cols = ['_id', '_smiles', '_mol']
    if grp_col: cols.append('_grp')
    cols.extend(task_cols)

    df = df.loc[good_ids, cols]

    task_labels = torch.from_numpy(df[task_cols].values).float()
    mask_missing = np.isfinite(task_labels).float()

    # Put -1 / -2 into mask to indicate > / < qualified values
    mask_missing[mask_greater.loc[good_ids, :].values] = -1
    mask_missing[mask_smaller.loc[good_ids, :].values] = -2

    return df, task_labels, mask_missing
