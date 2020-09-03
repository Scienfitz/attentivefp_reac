from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import MolStandardize
import logging
logger = logging.getLogger(__name__)


def chirality(atom):
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


def smiles2mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except:
        return None


def mol2smiles(mol):
    try:
        return Chem.MolToSmiles(mol)
    except:
        return None


def isEmptyMolecule(mol):
    if isinstance(mol, Chem.rdchem.Mol) and mol.GetNumAtoms() > 0:
        return False
    else:
        return True


standardizer = MolStandardize.Standardizer(prefer_organic=True)
def standardize_mol(mol):
    try:
        return standardizer.charge_parent(mol)
    except:
        return None
