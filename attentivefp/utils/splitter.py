from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem
from sklearn import model_selection
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)

def ClusterFps(fps, cutoff=0.4):
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1 - x for x in sims])
    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return cs

def ButinaCluster(mols, cutoff=0.4):
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in mols]
    clusters = [(cluster, [c_idx]*len(cluster)) for c_idx, cluster in enumerate(ClusterFps(fps, cutoff=cutoff))]
    clusterIDs = pd.DataFrame(np.hstack(clusters).transpose(), columns=['idx','cluster']).sort_values('idx')['cluster'].values
    return clusterIDs

def getSplit(mols, y, split=None, n_splits=5, groups=None, random_state=None):
    kv = None
    if split == 'KFold':
        kv = model_selection.KFold(n_splits=n_splits, random_state=random_state, shuffle=True).split(y)
    elif split == 'Random':
        # get the actual lists to ensure same are used for all evaluations
        kv = model_selection.ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=random_state).split(y)
    elif split == 'Butina':
        logger.info('Calculating Butina Clusters')
        clusters = ButinaCluster(mols, cutoff=0.8)
        kv =  model_selection.GroupKFold(n_splits=n_splits).split(y, groups=clusters)
    elif split == 'Scaffold':
        from rdkit.Chem import MolToSmiles
        from rdkit.Chem.Scaffolds import MurckoScaffold
        logger.info('Calculating Murcko Scaffolds')
        scaffolds = [MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))) for mol in mols]
        kv =  model_selection.GroupKFold(n_splits=n_splits).split(y, groups=scaffolds)
    elif split == 'GroupKFold':
        assert groups is not None, 'Please specify a splitColumn containing the groups when using "GroupKFold" split'
        assert len(groups) == len(y), 'Number of group labels do not match y data'
        kv =  model_selection.GroupKFold(n_splits=n_splits).split(y, groups=groups)
    elif split == 'Predefined':
        assert groups is not None, 'Please specify a splitColumn when using "Predifined" split'
        assert len(groups) == len(y), 'Number of labels do not match y data'
        kv = model_selection.LeaveOneGroupOut().split(y, groups=groups)
    elif split == 'Cumulative':
        assert groups is not None, 'Please specify a splitColumn when using "Cumulative" split'
        assert len(groups) == len(y), 'Number of labels do not match y data'
        kv = CummulativeSplit().split(y, groups=groups)
    elif split == 'None':
        kv = []
    else:
        logger.warning(f'Unknown split type {split}')
        raise ValueError(f'Unknown split type {split}')

    return list(kv)

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_array, _num_samples
from sklearn.utils import indexable
class CummulativeSplit(BaseCrossValidator):
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)
        return len(np.unique(groups))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        u = np.unique(groups)
        train_mask = groups == u[0]
        for i in u[1:]:
            test_mask = groups == i
            train_index = indices[train_mask]
            test_index = indices[test_mask]
            train_mask = train_mask | test_mask
            yield train_index, test_index
