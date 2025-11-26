import os
import pickle
import shutil

import polars as pl
import torch
from molecular_graph import molecular_graph_with_smiles
from torch_geometric.data import Dataset


class SMILESDataset(Dataset):
    """Molecular graph dataset with xyz as node features."""

    def __init__(
        self,
        root: str,
        raw_file: str,
        args_dict: dict,
        label_dict: dict,
        file_names: str = None,
        transform=None,
        pre_transform=None,
    ) -> None:
        """Initilizes the object.

        Parameters
        ----------
        root: str
            dataset storage (raw_dir and processed_dir)
        raw_file: str
            csv file contain the raw data
        args_dict: dict of any
            dictionary containing the parameters needed to create the molecualr graph
        label_dict: dict of int
            dictionary containing the labels to token mapping
        file_names: str
            default=None, file names for each molecular graph
        transform: object
            for torch_geometric.data.Dataset - not necessary in current case so ignored
        pre_transform: object
            for torch_geometric.data.Dataset - not necessary in current case so ignored

        Return
        ------
        None

        """
        self.file_names = file_names
        self.raw_file = raw_file
        self.args_dict = args_dict
        self.label_dict = label_dict
        super(SMILESDataset, self).__init__(root, transform, pre_transform)

    def raw_file_names(self):
        """If this file exist the dataset will not download from pytorch datasets."""
        return self.raw_file

    def processed_file_names(self):
        """If these file are found in raw_dir, processing is skipped."""
        if self.file_names == None:
            return "not_implemented.pt"
        else:
            return self.file_names

    def download(self):
        """Used if using a premade dataset."""
        pass

    def process(self):
        """Process the molecular graphs into a dataset."""
        self.data = pl.read_csv(self.raw_paths[0])
        smiles = self.data["smile"].to_list()
        labels = self.data["target"].to_list()

        coord = self.args_dict["coord"]

        for i, (smile, label) in enumerate(zip(smiles, labels)):
            temp_data = molecular_graph_with_smiles(
                smile, label, self.label_dict, coord
            )
            torch.save(temp_data, os.path.join(self.processed_dir, f"data_{i:06d}.pt"))

    def len(self):
        """Len of file names."""
        if self.file_names == None:
            self.data = pl.read_csv(self.raw_paths[0])
            smiles = self.data["smile"].to_list()
            return len(smiles)
        else:
            return len(self.file_names)

    def get(self, idx: int) -> object:
        """Get the molecular graphs.

        Paramaters
        ----------
        idx: int
            molecular graph index

        Return:
        ------
        data: object
            loaded in molecular graphs

        """
        data = torch.load(
            os.path.join(self.processed_dir, f"data_{idx:06d}.pt"), weights_only=False
        )
        return data

    def name(self, idx: int) -> str:
        """Get the name of the molecule at index idx.

        Paramaters
        ----------
        idx: int
            molecular graph index

        Return:
        ------
        name: str
            name of the molecule

        """
        data = torch.load(
            os.path.join(self.processed_dir, f"data_{idx:06d}.pt"), weights_only=False
        )
        return data.name
