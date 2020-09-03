import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.utils import mol_to_bigraph
import numpy as np

from ..utils.chem import isEmptyMolecule, smiles2mol, mol2smiles

import logging
logger = logging.getLogger(__name__)

class DGLFeaturizer(object):
    def __init__(self, device, AtomFeaturizer, BondFeaturizer):
        self.atom_featurizer = AtomFeaturizer
        self.bond_featurizer = BondFeaturizer
        self.device = device

    def _featurize_mol(self, mol):
        """Encodes mol as a DGL Graph object."""

        graph = None
        if not isEmptyMolecule(mol):
            try:
                graph = mol_to_bigraph(mol, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
                graph = graph.to(self.device)
            except Exception:
                logger.warning(f'Graph featurization failed for {mol2smiles(mol)}')
        return graph

    def featurize(self,
                  smiles_list
                  ):
        """Encodes a list of smiles as a DGL Graph object."""

        mols = [smiles2mol(s) for s in smiles_list]

        missing_mols = np.where([mol is None for mol in mols])[0]
        if len(missing_mols) > 0:
            missing_smiles = [str(smiles_list[i]) for i in missing_mols]
            logger.info(f'Smiles failed to convert into mol: {", ".join(missing_smiles)}')

        return self.featurize_mols(mols)

    def featurize_mols(self,
                       mol_list):
        """Encodes a list of rdkit mols as a DGL Graph object."""

        graphs = [self._featurize_mol(mol) for mol in mol_list]
        graphs = np.asarray(graphs)

        return graphs
