from dgllife.utils import BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer
from dgllife.utils import atom_type_one_hot, atom_degree_one_hot,                   \
                          atom_implicit_valence_one_hot, atom_formal_charge,        \
                          atom_num_radical_electrons, atom_hybridization_one_hot,   \
                          atom_is_aromatic, atom_total_num_H_one_hot,               \
                          atom_is_chiral_center, atom_chirality_type_one_hot
from functools import partial
                 
def atom_is_reactive(atom):
    """Get whether the atom is reactive. 
    This is identified with having the atom mapping number 1.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.

    """
    return [atom.GetAtomMapNum() == 1]
    
    
    
class ReactionAtomFeaturizer(BaseAtomFeaturizer):
    """The same featurizer as CanonicalAtomFeaturizer plut identification of the reactive site

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
        
    >>> # Get feature size for nodes
    >>> print(atom_featurizer.feat_size('feat'))
    75
    """
    def __init__(self, atom_data_field='h'):
        super(ReactionAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_reactive               #new
                 ]
            )})
            
            
class ReactionAttentiveFPAtomFeaturizer(BaseAtomFeaturizer):
    """The atom featurizer used in AttentiveFP plus identification of the reactive site
    AttentiveFP is introduced in
    `Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
    Attention Mechanism. <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__
    """
    def __init__(self, atom_data_field='h'):
        super(ReactionAttentiveFPAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [partial(atom_type_one_hot, allowable_set=[
                    'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S',
                    'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'], encode_unknown=True),
                 partial(atom_degree_one_hot, allowable_set=list(range(6))),
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 partial(atom_hybridization_one_hot, encode_unknown=True),
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_is_chiral_center,
                 atom_chirality_type_one_hot,
                 atom_is_reactive               #new
                 ]
            )})
