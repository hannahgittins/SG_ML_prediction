"""Hannah Gittins
Test accuracy of a model
"""

import os
import pickle
from datetime import datetime

import pandas as pd
import psutil
import torch
from MPNN_model import MP_model
from tools.graph_dataset import XYZDataset
from tools.utils import log_memory_usage, multi_accuracy, open_dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def remove_coord(dataset: object) -> object:
    """Remove coordinates from the data.

    Parameters
    ----------
    dataset: object
        test set

    Return
    ------
    dataset: object
        test set with no coordinates

    """
    new_dataset = []
    dataset = dataset.to_data_list()
    for data in dataset:
        new_data = data  # remove the first three columns (x, y, z coordinates)
        new_data.x = data.x[:, 0:5]
        new_dataset.append(new_data)

    assert new_dataset[0].x.shape[1] == 5, (
        "The node features should have 5 columns after removing coordinates."
    )

    print(f"new dataset node features size:", new_dataset[0].x.shape[1])

    return Batch.from_data_list(new_dataset)


def test_model(
    model_file_path: str,
    label_file_path: str,
    mp_layers: int,
    edge_layers: int,
    fcc_in_gnn: int,
    graphnorm: bool,
    hidden: int,
    test_dataset: object,
    device: str,
    output: str,
):
    """Test gnn model.

    Parameters
    ----------
    model_file_path: str
        file path to the model of interest
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
    device: str
        device (cuda or cpu)
    output: str
        results output filename

    Return
    ------
    None

    """
    test_loader = DataLoader(
        test_dataset, batch_size=32, pin_memory=False, shuffle=False
    )

    with open(label_file_path, "rb") as f:
        label_dict = pickle.load(f)

    log_memory_usage("Before loading model")

    example = test_dataset[0]
    model = MP_model(
        num_node_features=(example.num_node_features - 3),
        num_classes=len(label_dict),
        num_edge_features=(example.num_edge_features),
        mp_layers=mp_layers,
        edge_layers=edge_layers,
        fcc_in_gnn=fcc_in_gnn,
        graphnorm=graphnorm,
        hidden=hidden,
    )
    model = model.to(device)

    log_memory_usage("After loading model")

    final_checkpoint = torch.load(
        model_file_path, weights_only=False, map_location=torch.device(device)
    )
    model.load_state_dict(final_checkpoint["model_state_dict"])
    model.eval()
    log_memory_usage("Before loop")

    predictions = []
    targets = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            log_memory_usage("Loaded data in loop")
            print(data.size())
            data = data.to(device)
            data = remove_coord(data)
            print(data.size())
            out = model(
                data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
            )
            log_memory_usage("Calculated output")

            print(out.cpu())

            predictions.extend(out.cpu())
            targets.append(data.y.cpu())

    print(predictions)

    predictions = torch.stack(predictions)
    targets = torch.cat(targets)

    acc_dict = {}
    for i in [1, 3, 5, 10]:
        acc_dict[i] = multi_accuracy(
            predictions, y_test=targets, label_dict=label_dict, N=i
        )
    log_memory_usage("Calculated accuracy")

    with open(output, "w") as f:
        f.write(f"Accuracy for top 1: {acc_dict[1] * 100:.4f}\n")
        f.write(f"Accuracy for top 3: {acc_dict[3] * 100:.4f}\n")
        f.write(f"Accuracy for top 5: {acc_dict[5] * 100:.4f}\n")
        f.write(f"Accuracy for top 10: {acc_dict[10] * 100:.4f}\n")
    log_memory_usage("Wrote output to file")


if __name__ == "__main__":
    log_memory_usage("Start")
    ## turn into a function!!
    device = "cpu"
    file_num = "01"
    mp_layers = 3
    edge_layers = 1
    fcc_in_gnn = 2
    hidden = 32
    graphnorm = True
    label_file_path = "/scratch/hs9g19/GNN_store/datasets/All_organics_opt_molecular_graphs_coord_label_dict.pickle"
    model_file_path = (
        f"/scratch/hs9g19/GNN_store/no_coord/best_gnn_model_ddp_test_{file_num}.pt"
    )
    output = f"/scratch/hs9g19/GNN_store/no_coord/best_gnn_model_{file_num}_output.txt"

    # folder = "/scratch/hs9g19/GNN_store/datasets/"
    # # file = "EVEN_opt_molecular_coord_"

    storage_folder = "/scratch/hs9g19/GNN_store/datasets"
    front = "Unbalanced_opt_no_polymorphs"

    with open(label_file_path, "rb") as f:
        label_dict = pickle.load(f)

    args_dict = {
        "xyz_folder": "/scratch/hs9g19/XYZ_files",
        "mol_folder": "/scratch/hs9g19/Mol_files_with_Hs/Mol_files_with_Hs",
        "opt": True,
        "coord": True,
        "all_dist": False,
        "second_neighbor": False,
    }

    test_dataset = open_dataset(storage_folder, front, "test", label_dict, args_dict)
    log_memory_usage("After loading dataset")

    test_model(
        model_file_path,
        label_file_path,
        mp_layers,
        edge_layers,
        fcc_in_gnn,
        graphnorm,
        hidden,
        test_dataset,
        device,
        output,
    )
    log_memory_usage("After testing model")
