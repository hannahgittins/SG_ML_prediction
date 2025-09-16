"""Hannah Gittins
Test accuracy of a model
"""

import os
import pickle

import pandas as pd
import torch
from EGNN_edit_model import EGNN_edit
from tools.utils import multi_accuracy
from torch_geometric.loader import DataLoader


def test_model(
    model_file_path: str,
    label_file_path: str,
    mp_layers: int,
    fcc_in_gnn: int,
    test_dataset: object,
    device: str,
    output: str,
    num_classes: int,
) -> None:
    """Test egnn model.

    Parameters
    ----------
    model_file_path: str
        file path to the model of interest
    label_file_path: str
        file path to the label dictionary (.pkl file)
    mp_layers: int
        number of message passing layers
    fcc_in_gnn: int
        number of nn layers post gnn
    test_dataset: object
        test dataset
    device: str
        device (cuda or cpu)
    output: str
        test accuracy output
    num_classes: str
        number of targets

    Return
    ------
    None

    """
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_dataset), pin_memory=True, shuffle=False
    )
    print("debugging: test set loaded")

    with open(label_file_path, "rb") as f:
        label_dict = pickle.load(f)

    example = test_dataset[0]
    model = EGNN_edit(example, num_classes, mp_layers, fcc_in_gnn)
    model = model.to(device)

    final_checkpoint = torch.load(
        model_file_path, weights_only=False, map_location=torch.device(device)
    )
    model.load_state_dict(final_checkpoint["model_state_dict"])
    model.eval()
    print("debugging: model loaded")
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)

        print("debugging: data loaded")

        acc_dict = {}
        for i in [1, 3, 5, 10]:
            acc_dict[i] = multi_accuracy(out, y_test=data.y, label_dict=label_dict, N=i)

        with open(output, "w") as f:
            f.write(f"Accuracy for top 1: {acc_dict[1] * 100:.4f}\n")
            f.write(f"Accuracy for top 3: {acc_dict[3] * 100:.4f}\n")
            f.write(f"Accuracy for top 5: {acc_dict[5] * 100:.4f}\n")
            f.write(f"Accuracy for top 10: {acc_dict[10] * 100:.4f}\n")
        print("debugging: printed outpit")


if __name__ == "__main__":
    ## turn into a function!!
    device = "cpu"
    file_num = "08"
    mp_layers = 3
    fcc_in_gnn = 2
    label_file_path = "/scratch/hs9g19/GNN_store/datasets/All_organics_opt_molecular_graphs_coord_label_dict.pickle"
    model_file_path = f"/scratch/hs9g19/GNN_store/EGNN/best_model_test_{file_num}.pt"
    output = f"/scratch/hs9g19/GNN_store/EGNN/best_model_test_{file_num}_output.txt"

    # folder = "/scratch/hs9g19/GNN_store/datasets/"
    # # file = "EVEN_opt_molecular_coord_"

    storage_folder = "/scratch/hs9g19/GNN_store/datasets"
    front = "Unbalanced_opt_no_polymorphs"

    with open(label_file_path, "rb") as f:
        label_dict = pickle.load(f)

    num_classes = len(label_dict)

    args_dict = {
        "xyz_folder": "/scratch/hs9g19/XYZ_files",
        "mol_folder": "/scratch/hs9g19/Mol_files_with_Hs/Mol_files_with_Hs",
        "opt": True,
        "coord": True,
        "all_dist": False,
        "second_neighbor": False,
    }

    test_dataset = open_dataset(storage_folder, front, "test", label_dict, args_dict)
    print("here1")

    test_model(
        model_file_path,
        label_file_path,
        mp_layers,
        fcc_in_gnn,
        test_dataset,
        device,
        output,
        num_classes,
    )
    print("here2")
