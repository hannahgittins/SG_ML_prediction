"""Script used to create molecular graphs"""

import os
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Chem.rdchem import BondType, HybridizationType
from rdkit.Chem.rdmolops import AddHs
from rdkit.Chem.rdMolTransforms import (
    CanonicalizeConformer,
    GetBondLength,
)
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# import ccdc
# from ccdc import io


def one_hot_matrix(inputs: list, feature_dict: dict):
    """Function that one-hot encodes the features and returns it as a matrix

    Parameters
    ----------
    input: list
        the list of input values that need to be converted to one-hot values
    feature_dict: dict
        the dictionary used to make sure all the matrices are the same

    Returns
    -------
    one_hot: np.array
        the one-hot encoded features as a matrix

    """
    value = feature_dict[inputs]
    one_hot = np.eye(len(feature_dict))[value]
    return one_hot


def bonds(
    mol: object,
    bonds_list: list,
    swap_type_dict: dict,
    length: bool = False,
    dist: bool = False,
    second_neighbor: bool = False,
    atom_idx=None,
):
    """Returns the node-node connections (bonds) as well as caculating the bond features

    Parameters
    ----------
    mol : object
        The rdkit mol object for the molecule of interest.
    bonds_list : list
        A list of the bonds in the molecule of interest (determined in the atom function).
    swap_type_dict : dict
        A dictionary containing bond types.
    length : bool, optional
        Whether to include bond length of bonded atoms, by default False.
    dist : bool, optional
        Whether to include the distance matrix (all atom-atom distances), by default False.
    second_neighbor : bool, optional
        Whether to include bond length of bonded atoms and second neighbor atoms, by default False.
    atom_idx : list, optional
        List of atom indices, required if dist or second_neighbor is True, by default None.

    Returns
    -------
    A tuple containing the following bond features:
    - node_node : np.array
        The two nodes (indexes) that are bonded together.
    - bond_type : np.array
        The bond type (single, 1, aromatic, 1.5, double, 2, triple, 3 etc).
    - conjugate : np.array
        If the bond is conjugated or not (binary).
    - ring : np.array
        If the bond is a ring or not (binary).
    - length : np.array, optional
        The length of the bond in pm (if length, dist, or second_neighbor is True).

    """

    def _add_bond(a: int, b: int) -> None:
        """Used to add a bond between a and b (where a and b are the indexes of the atoms)"""
        non_bonded.add((a, b))
        non_bonded.add((b, a))
        atom_index["start"].extend([a, b])
        atom_index["end"].extend([b, a])
        bond_features["bond type"].extend([0, 0])
        bond_features["conjugate type"].extend([False, False])
        bond_features["ring value"].extend([False, False])

        pos_a = np.array(conf.GetAtomPosition(a))
        pos_b = np.array(conf.GetAtomPosition(b))
        bond_length.extend(
            [np.linalg.norm(pos_a - pos_b), np.linalg.norm(pos_a - pos_b)]
        )

    atoms_temp = {"start": [], "end": []}

    bond_features = {"bond type": [], "conjugate type": [], "ring value": []}

    for bond in bonds_list:
        bond_features["bond type"].append(swap_type_dict[bond.GetBondType()])
        bond_features["conjugate type"].append(int(bond.GetIsConjugated()))
        bond_features["ring value"].append(int(bond.IsInRing()))

        atoms_temp["start"].append(bond.GetBeginAtomIdx())
        atoms_temp["end"].append(bond.GetEndAtomIdx())

    bond_features["bond type"] *= 2
    bond_features["conjugate type"] *= 2
    bond_features["ring value"] *= 2

    atom_index = {
        "start": list(atoms_temp["start"] + atoms_temp["end"]),
        "end": list(atoms_temp["end"] + atoms_temp["start"]),
    }

    if length or dist or second_neighbor:
        conf = mol.GetConformer(0)
        CanonicalizeConformer(
            conf
        )  ## centers the molecule and aligns with the principal axes
        bond_length = [
            GetBondLength(conf, int(start), int(end))
            for start, end in zip(atom_index["start"], atom_index["end"])
        ]

    if dist:
        non_bonded = set()
        pairs = set(zip(atom_index["start"], atom_index["end"]))
        for i in atom_idx:
            for j in atom_idx:
                if i == j and ((i, j) in non_bonded) or ((i, j) in pairs):
                    continue
                elif (i, j) not in pairs:
                    _add_bond(i, j)

    node_node = np.stack(
        [np.array(atom_index["start"]), np.array(atom_index["end"])], axis=0
    )

    if length or second_neighbor or dist:
        return (
            node_node,
            np.array(bond_features["bond type"]),
            np.array(bond_features["conjugate type"]),
            np.array(bond_features["ring value"]),
            np.array(bond_length),
        )
    else:
        return (
            node_node,
            np.array(bond_features["bond type"]),
            np.array(bond_features["conjugate type"]),
            np.array(bond_features["ring value"]),
        )


def atoms(mol: object, hybrid_dict: dict, coord: bool = False):
    """Returns the features for the atoms for the molecule of interest
    Parameters:
    mol : object
        The rdkit mol object for the molecule of interest.
    hybrid_dict : dict
        A dictionary that maps hybridization types to numerical values.
    coord : bool, optional
        Whether to include atomic coordinates, by default False.

    Returns:
    A tuple containing the following atom features:
    - atomic number : np.array
        The atomic numbers of the atoms.
    - hybridization : np.array
        The hybridization states of the atoms.
    - charge : np.array
        The formal charges of the atoms.
    - aromatic : np.array
        Binary values indicating if the atoms are aromatic.
    - partial : np.array
        The Gasteiger partial charges of the atoms.
    - idx : list
        The indices of the atoms.
    - bond_list : list
        A list of bonds in the molecule.
    - coord_x : np.array, optional
        The x-coordinates of the atoms (if coord is True).
    - coord_y : np.array, optional
        The y-coordinates of the atoms (if coord is True).
    - coord_z : np.array, optional
        The z-coordinates of the atoms (if coord is True).

    """
    atom_list = mol.GetAtoms()

    ComputeGasteigerCharges(mol)
    # print(atom_list)

    idx = []
    bond_list = []
    atom_features = {
        "atomic number": [],
        "hybridization": [],
        "charge": [],
        "aromatic": [],
        "partial": [],
    }

    for atom in atom_list:
        idx.append(atom.GetIdx())
        bonds_temp = atom.GetBonds()
        temp = [bond for bond in bonds_temp]
        bond_list.extend(temp)

        atom_features["atomic number"].append(atom.GetAtomicNum())
        atom_features["hybridization"].append(hybrid_dict[atom.GetHybridization()])
        atom_features["charge"].append(atom.GetFormalCharge())
        atom_features["aromatic"].append(int(atom.GetIsAromatic()))

        gast = float(atom.GetProp("_GasteigerCharge"))
        if np.isnan(gast) or np.isinf(gast):
            gast = 0.0

        atom_features["partial"].append(gast)

    bond_list = list(set(bond_list))

    if coord:
        mol = Chem.AddHs(mol)  # Add hydrogens if needed

        # Generate conformer
        ret = AllChem.EmbedMolecule(mol)
        if ret != 0:
            print("Embedding failed for molecule:")

        conf = mol.GetConformer(0)
        CanonicalizeConformer(
            conf
        )  ## centers the molecule and aligns with the principal axes
        # coord = np.zeros((len(idx), 3))
        coord_x = []
        coord_y = []
        coord_z = []
        for atom in idx:
            atomic_pos = conf.GetAtomPosition(atom)
            x, y, z = atomic_pos.x, atomic_pos.y, atomic_pos.z
            coord_x.append(x)
            coord_y.append(y)
            coord_z.append(z)
            # coord[atom][:] = [x, y, z]

    if not coord:
        return (
            np.array(atom_features["atomic number"]),
            np.array(atom_features["hybridization"]),
            np.array(atom_features["charge"]),
            np.array(atom_features["aromatic"]),
            np.array(atom_features["partial"]),
            idx,
            bond_list,
        )
    else:
        return (
            np.array(atom_features["atomic number"]),
            np.array(atom_features["hybridization"]),
            np.array(atom_features["charge"]),
            np.array(atom_features["aromatic"]),
            np.array(atom_features["partial"]),
            idx,
            bond_list,
            np.array(coord_x),
            np.array(coord_y),
            np.array(coord_z),
        )


def molecular_graph_with_smiles(
    smile: str, label: int, label_dict: dict, coord: bool = False
):
    """Used to return a molecular graph
    Parameters:
    smile : str
        The SMILES string of the molecule.
    label : int
        The target label for the molecule.
    label_dict : dict
        A dictionary to convert the label to a tokenized value.
    coord : bool, optional
        Whether to include atomic coordinates in the graph, by default False.

    Returns:
    Data
        A PyTorch Geometric Data object representing the molecular graph.

    """
    hybrid_values = HybridizationType.values
    swap_hybrid_dict = {value: key for key, value in hybrid_values.items()}

    bond_type_dict = BondType.values
    swap_type_dict = {value: key for key, value in bond_type_dict.items()}

    mol = MolFromSmiles(smile)
    mol = AddHs(mol)

    if coord == False:
        atom_nums, hybrid, charge, aromatic, partial, _, bonds_list = atoms(
            mol, swap_hybrid_dict
        )
        x = np.stack([atom_nums, hybrid, charge, aromatic, partial], axis=1)
    elif coord == True:
        atom_feat = atoms(mol, swap_hybrid_dict, coord=True)
        atom_nums, hybrid, charge, aromatic, partial, atom_idx, bonds_list = atom_feat[
            :7
        ]
        coord_x, coord_y, coord_z = atom_feat[7:10]
        x = np.stack(
            [atom_nums, hybrid, charge, aromatic, partial, coord_x, coord_y, coord_z],
            axis=1,
        )

    node_node, bond_type, conjugate_type, ring_value = bonds(
        mol, bonds_list, swap_type_dict
    )
    edge_attr = np.stack([bond_type, conjugate_type, ring_value], axis=1)

    label = label_dict[label]

    return Data(
        x=torch.Tensor(x),
        edge_index=torch.Tensor(node_node).long(),
        edge_attr=torch.Tensor(edge_attr),
        y=label,
        name=str(smile),
    )


if __name__ == "__main__":
    label_dict = {1: 0, 2: 1, 3: 2}

    # temp = molecular_graph_with_xyz("AACMHX10", xyz_folder="C:\\Users\\hs9g19\\Documents\\PhD\\MainQuest\\SpaceGroupProject\\SpaceGroupData\\Data",
    #                                 mol_folder="C:\\Users\\hs9g19\\Documents\\PhD\\MainQuest\\SpaceGroupProject\\SpaceGroupData\\Data",
    #                                 label=2, label_dict=label_dict, second_neighbor=True) #
    # print(temp.is_undirected())
    # # print(temp.edge_index)
    # # print(temp.edge_attr)

    temp = molecular_graph_with_smiles("CC(N)C(O)=O", label_dict=label_dict, label=2)

    net_temp = to_networkx(temp, to_undirected=True)
    nx.draw(net_temp, with_labels=True)
    plt.savefig("alanine_example.png")

    # pass
    # example = molecular_graph("c1ccccc1")

    # print(f"Nodes: {example.x}")
    # print(f"Edges: {example.edge_index}")

    # temp = molecular_graph_for_visnet("ABAHET", "C:\\Users\\hs9g19\\Documents\\PhD\\MainQuest\\SpaceGroupProject\\SpaceGroupData", 1, label_dict, opt=True)

    # print(temp.pos)
    # print(temp.z)
