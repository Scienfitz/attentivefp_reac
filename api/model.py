import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json

import torch

from attentivefp.models.dgl import EnsembleAttFP, AttentiveFPDense
from attentivefp.models.training import predict as att_predict
from attentivefp.featurizer.graph import DGLFeaturizer
from attentivefp.models.training import predict as att_predict
from attentivefp.utils.chem import standardize_mol, smiles2mol, mol2smiles

import logging
logger = logging.getLogger(__name__)

available_models = {'marco': Path(__file__).parent / 'models' / 'marco', 'dgl': Path(__file__).parent / 'models' / "dgl"}

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

torch.set_num_threads(1)

class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self._featurizer = DGLFeaturizer(device)
        self.columns = None
        self.load()

    def predict(self, mols: np.ndarray) -> np.ndarray:
        graphs = self._featurizer.featurize_mols(mols)

        bad_idx = np.where(np.equal(graphs, None))[0]
        good_idx = np.where(np.not_equal(graphs, None))[0]
        graphs = graphs[good_idx]

        preds, std = att_predict(graphs, self._model, device, batch_size=100, dropout_samples=0)
        att_df = pd.DataFrame(np.concatenate([preds, std], axis=1), index=good_idx, columns=self.columns + [f'{c}:UNCERT' for c in self.columns])
        att_df = att_df.reindex(range(len(mols)), axis=0).reindex(sorted(att_df.columns), axis=1)

        att_df.insert(loc=0, column="IDX", value=att_df.index)
        if len(bad_idx) > 0:
            att_df.loc[att_df.index[bad_idx],'error'] = 'Mol conversion failed'

        return att_df

    def load(self):
        try:
            with open(self._model_path / 'model.json', 'r') as f:
                _model_json = json.load(f)
                self.columns = _model_json['columns']
            self._model = torch.load(self._model_path / 'model.pth', map_location=device).to(device)
        except:
            logger.exception('model load failed')
            self._model = None
        return self

def get_model(model_key):
    logger.info(f'get_model {model_key}')
    return model_cache.get(model_key, None)

def get_columns(model_key):
    m = model_cache.get(model_key, None)
    if m is not None:
        return m.columns
    else:
        return None

def avail():
    return [(k, get_columns(k)) for k in model_cache.keys()]

# pre-load all models
model_cache = {}
for a, model_path  in available_models.items():
    model = Model(model_path)
    model_cache[a] = model
    logger.info(f'loaded model {a} from {model_path}')