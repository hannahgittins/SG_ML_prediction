"""Prediciton of space group from GNN_1 -> GNN without coordinates."""

import os
import pickle
import sys
from typing import Union

import pandas as pd
import torch
from graph_dataset import SMILESDataset
from torch_geometric.loader import DataLoader

sys.path.append("..\\")
from gnn.MPNN_model import MP_model


def N_predicitons(out: torch.Tensor, label_dict: dict, N: int = 10) -> float:
    """Return the space group predictions for top N.

    Parameters
    ----------
    out: torch.Tensor
        output of the GNN prediciton
    label_dict: dict
        dictionary containing the true value target (space group) and the given label
    N: int
        top N, default=10

    Return
    ------
    N_predicted: list of int
        list of top N predicted space groups

    """
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    prob = torch.nn.functional.softmax(out, dim=1)

    _, top_n_preds = prob.topk(N, dim=1)
    top_n_preds = top_n_preds.tolist()

    N_predicted = [reverse_label_dict[pred] for pred in top_n_preds[0]]

    return N_predicted


def test_model(
    test_dataset: object,
    coord: bool = False,
    device: str = "cpu",
    label_file_path: str = "label_dict\\All_organics_opt_molecular_graphs_coord_label_dict.pickle",
    mp_layers: int = 3,
    edge_layers: int = 1,
    fcc_in_gnn: int = 2,
    graphnorm: bool = True,
    hidden: int = 32,
):
    """Test gnn model.

    Parameters
    ----------
    test_dataset: object
        test dataset
    device: str
        device (cuda or cpu)
    coord: bool
        whether coordinates were used in training
    label_file_path: str
        file path to the label dictionary (.pkl file)
    mp_layers: int
        number of message passing layers
    edge_layers: int
        number of edge layers
    fcc_in_gnn: int
        number of nn layers post gnn
    graphnorm: bool
        graph normalisation
    hidden: int
        number of hidden layers
    test_dataset: object
        test dataset

    Return
    ------
    None

    """
    if coord:
        model_file_path = "models\\gnn_2_coord_model.pt"
    else:
        model_file_path = "models\\gnn_1_model.pt"

    test_loader = DataLoader(
        test_dataset, batch_size=1, pin_memory=False, shuffle=False
    )

    with open(label_file_path, "rb") as f:
        label_dict = pickle.load(f)

    example = test_dataset[0]
    model = MP_model(
        num_node_features=(example.num_node_features),
        num_classes=len(label_dict),
        num_edge_features=(example.num_edge_features),
        mp_layers=mp_layers,
        edge_layers=edge_layers,
        fcc_in_gnn=fcc_in_gnn,
        graphnorm=graphnorm,
        hidden=hidden,
    )
    model = model.to(device)

    final_checkpoint = torch.load(
        model_file_path, weights_only=False, map_location=torch.device(device)
    )
    model.load_state_dict(final_checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(
                data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
            )

            predictions = out.cpu()

            top_10_preds = N_predicitons(predictions, label_dict)

            print(f"{data.name} top 10 predicted space groups:")
            for pred in top_10_preds:
                print(f"  {pred}")


def make_dataset(
    smiles: Union[list[str], str],
    working_directory: str,
    coord: bool = False,
    label_file_path: str = "label_dict\\All_organics_opt_molecular_graphs_coord_label_dict.pickle",
) -> object:
    """Make dataset from smiles list.

    Parameters
    ----------
    smiles: list or str
        list of smiles or single smile string
    working_directory: str
        working directory to store temporary files
    coord: bool
        whether to use coordinates
    label_file_path: str
        file path to the label dictionary (.pkl file)

    Return
    ------
    dataset: object
        torch dataset object

    """
    if isinstance(smiles, str):
        smiles = [smiles]

    with open(label_file_path, "rb") as f:
        label_dict = pickle.load(f)

    process = os.path.join(working_directory, "processed")
    raw = os.path.join(working_directory, "raw")

    if os.path.isdir(working_directory) == False:
        os.mkdir(working_directory)
        os.mkdir(process)
        os.mkdir(raw)
    elif os.path.isdir(process) == False:
        os.mkdir(process)
    elif os.path.isdir(raw) == False:
        os.mkdir(raw)

    ## Make raw csv file:
    raw_filepath = os.path.join(working_directory, "raw", "temp_smiles.csv")
    df = pd.DataFrame(smiles, columns=["smile"])
    df["target"] = [1] * len(smiles)  ## Dummy target values - not used in prediction
    df.to_csv(raw_filepath, index=False)

    args_dict = {
        "coord": coord,
    }
    file_names = None

    dataset = SMILESDataset(
        root=working_directory,
        raw_file="temp_smiles.csv",
        file_names=file_names,
        label_dict=label_dict,
        args_dict=args_dict,
    )

    return dataset


if __name__ == "__main__":
    coord = False
    smiles = ["C1=CC=CC=C1", "C1=CC=CN=C1"]
    temp_dir = "temp_gnn_1_prediction"

    dataset = make_dataset(
        smiles=smiles,
        working_directory=temp_dir,
        coord=coord,
    )

    test_model(test_dataset=dataset, coord=coord)
