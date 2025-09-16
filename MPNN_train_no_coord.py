"""Train model with coordinates removed from molecular graph."""

import os
import pickle
import shutil

import numpy as np
import polars as pl
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Batch

from MPNN_model import MP_model
from utils import (
    _count_open_files,
    load_checkpoint,
    open_dataset,
    prepare_dataloader,
    setup,
)


class Trainer:
    """Trains the model."""

    def __init__(
        self,
        model: object,
        train_data: object,
        val_data: object,
        lr: float,
        criterion: object,
        gpu_id: int,
        best_path: str,
        ckp_path: str,
        total_epochs: int,
    ):
        """Initialize the object and storage of values

        Parameters
        ----------
        model : object
            the gnn model (already set up)
        train_data : torch_geometertic.loader.DataLoader
            training data
        val_data : torch_geometertic.loader.DataLoader
            validation data
        lr : float
            learning rate
        criterion : object
            loss function
        gpu_id : int
            gpu id
        best_path : str
            filepath and name to store the best model
        ckp_path : str
            filepath and name to store the current model
        total_epochs : int
            total number of epochs

        Returns
        -------
        None

        """
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.gpu_id = gpu_id
        self.best_path = best_path
        self.checkpoint_path = ckp_path
        self.total_epochs = total_epochs

        df_path = ckp_path.split(".")[0]
        df_path += ".csv"
        self.df_path = df_path

        if os.path.isfile(df_path):
            df = pl.read_csv(df_path)
            loss_store = df.to_dict(as_series=False)
        else:
            loss_store = {
                "epoch": [],
                "val loss": [],
                "val accuracy": [],
                "train loss": [],
                "train accuracy": [],
                "gpu": [],
            }
        self.loss_store = loss_store

        if os.path.isfile(ckp_path) and os.path.isfile(best_path):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model, _, start_epoch, loss = load_checkpoint(model, optimizer, ckp_path)
            current_best_model = torch.load(best_path)
            self.start_epoch = start_epoch
            self.best_loss = current_best_model[
                "losslogger"
            ]  ## loading in current best loss
            print(
                f"Found checkpoint and best model - will start training from epoch {start_epoch} with loss of {loss}."
            )
        elif os.path.isfile(ckp_path) and not os.path.isfile(best_path):
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model, _, start_epoch, loss = load_checkpoint(model, optimizer, ckp_path)
            self.start_epoch = start_epoch
            self._save_best(start_epoch, loss)
            print(
                f"Found checkpoint - will start training from epoch {start_epoch} with loss of {loss}. \nCouldn't find best model - saving current checkpoint as the best."
            )
        else:
            self.start_epoch = 1
            self.best_loss = float("inf") if isinstance(best_path, str) else None
            print("Couldn't find checkpoint - will start training from beginning.")

        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def _remove_coord_batch(self, dataset: object):
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

        return Batch.from_data_list(new_dataset).to(self.gpu_id)

    def _run_batch(self, data: object):
        self.optimizer.zero_grad()

        data = self._remove_coord_batch(data)

        x = data.x.to(self.gpu_id, dtype=torch.float32)
        edge_index = data.edge_index.to(self.gpu_id)
        edge_attr = data.edge_attr.to(self.gpu_id)
        batch = data.batch.to(self.gpu_id)

        out = self.model(x, edge_index, edge_attr, batch)
        # print(out)
        loss = self.criterion(out, data.y.to(self.gpu_id))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

    def _run_epoch(self, epoch: int, dataset: object):
        self.model.train()
        print(f"GPU:{self.gpu_id} | Epoch {epoch}")
        self.train_data.sampler.set_epoch(epoch)
        for data in dataset:
            data.to(self.gpu_id)
            self._run_batch(data)

    def _eval_dataset(self, dataset: object):
        self.model.eval()
        total_correct = 0
        total_loss = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                # correct_log, loss = self._evaluate(data)
                data = self._remove_coord_batch(data).to(self.gpu_id)

                out = self.model(
                    data.x.to(dtype=torch.float32),
                    data.edge_index,
                    data.edge_attr.to(dtype=torch.float32),
                    data.batch,
                )
                print(f"out.shape: {out.shape}, y.shape: {data.y.shape}")
                loss = self.criterion(out.to(self.gpu_id), data.y.to(self.gpu_id))
                pred = out.argmax(dim=1)

                total_correct += (pred == data.y).sum().item()
                total_loss += loss.item() * data.size(0)
                total += data.y.size(0)

        total_loss = torch.tensor(total_loss, device=self.gpu_id)
        total_correct = torch.tensor(total_correct, device=self.gpu_id)
        total = torch.tensor(total, device=self.gpu_id)

        torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)

        if total.item() == 0:
            return 0, float("inf")

        avg_loss = total_loss.item() / total.item()
        accuracy = total_correct.item() / total.item()

        return accuracy, avg_loss

    def _save_best(self, epoch: int, val_loss: float):
        self.best_loss = val_loss
        print(f"Saving new model with loss of {val_loss} after {epoch} epoch(s)")
        shutil.copyfile(self.checkpoint_path, self.best_path)

    def _save_current(self, epoch: int, val_loss: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.criterion,
                "losslogger": val_loss,
            },
            self.checkpoint_path,
        )

        if isinstance(self.best_path, str) and val_loss < self.best_loss:
            self._save_best(epoch, val_loss)

    def train(self):
        """Run training loop over X epochs."""
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            self._run_epoch(epoch, self.train_data)

            torch.distributed.barrier()
            val_accuracy, val_loss = self._eval_dataset(self.val_data)
            train_accuracy, train_loss = self._eval_dataset(self.train_data)

            if self.gpu_id == 0:
                self._save_current(epoch, val_loss)
                print(
                    f"Epoch: {epoch}/{self.total_epochs} | Validation accuracy: {val_accuracy * 100:.4f} | Validation loss: {val_loss}"
                )
                print(
                    f"Train accuracy: {train_accuracy * 100:.4f} | Train loss: {train_loss}"
                )

                # Update loss store
                self.loss_store["epoch"].append(epoch)
                self.loss_store["val loss"].append(val_loss)
                self.loss_store["val accuracy"].append(val_accuracy)
                self.loss_store["train loss"].append(train_loss)
                self.loss_store["train accuracy"].append(train_accuracy)
                self.loss_store["gpu"].append(self.gpu_id)

                # Save to CSV
                temp = pl.DataFrame(self.loss_store)
                temp.write_csv(self.df_path)


def main(
    local_rank: int,
    world_size: int,
    total_epochs: int,
    batch_size: int,
    train_dataset: list,
    val_dataset: list,
    best_path: str,
    ckp_path: str,
    num_classes: int,
    weight: list,
    classic_dropout: float = 0,
    dropnode: float = 0,
    mp_layers: int = 3,
    edge_layers: int = 3,
    fcc_in_gnn: int = 5,
) -> None:
    """Prepares data and trains the model.

    Parameters
    ----------
    local_rank : int
        gpu id (ish)
    world_size : int
        number of gpus
    total_epochs : int
        total number of epochs
    batch_size : int
        batch size
    train_dataset : list of torch_geometric.data.Data
        training set - list of molecular graph graphs
    val_dataset : list of torch_geometric.data.Data
        validation set - list of molecular graph graphs
    best_path : str
        filepath to the best model storage
    ckp_path : str
        filepath to the current checkpoint
    num_classes : int
        number of targets
    weight: list
        loss weights
    classic_dropout: float
        default=0, nn dropout
    dropnode: float
        default=0
    mp_layers: int
        default=3, number of message passing layers
    edge_layers: int
        default=3, number of nn in the edge layer
    fcc_in_gnn: int
        default=5, number of fcc layers after gnn


    Returns
    -------
    None

    """
    setup(local_rank, world_size)

    print(
        f"Process {local_rank}/{world_size} is running on device: {torch.cuda.current_device()}"
    )

    train_data = prepare_dataloader(train_dataset, batch_size)
    val_data = prepare_dataloader(val_dataset, batch_size)

    example = train_dataset[0]
    print(
        f"Example -> edge_index {example.edge_index.size()}, x {example.x.size()}, edge_attr {example.num_edge_features}"
    )
    print(f"Example -> node features: {example.num_node_features - 3}")
    model = MP_model(
        num_node_features=(example.num_node_features - 3),
        num_classes=num_classes,
        num_edge_features=(example.num_edge_features),
        mp_layers=mp_layers,
        edge_layers=edge_layers,
        fcc_in_gnn=fcc_in_gnn,
        dropout=classic_dropout,
        dropnode=dropnode,
        graphnorm=True,
    )

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # weight = compute_class_weight('balanced', classes=classes, y=labels)

    criterion = (
        torch.nn.CrossEntropyLoss(weight=weight.to(local_rank))
        if weight is not None
        else torch.nn.CrossEntropyLoss().to(local_rank)
    )

    trainer = Trainer(
        model,
        train_data,
        val_data,
        0.0001,
        criterion,
        local_rank,
        best_path,
        ckp_path,
        total_epochs,
    )
    trainer.train()

    # when finished removes memory
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    total_epochs = int  # change these
    batch_size = int
    mp_layers = int
    edge_layers = int
    fcc_in_gnn = int
    dropnode = float
    classic_dropout = float
    best_path = str
    ckp_path = str
    target_dict_file = str

    storage_folder = str
    front = str

    # print(world_size)
    # print("Is GPU available for torch_geometric:", torch.cuda.is_available())

    _count_open_files()

    with open(target_dict_file, "rb") as f:
        label_dict = pickle.load(f)

    args_dict = {
        "xyz_folder": "/scratch/hs9g19/XYZ_files",
        "mol_folder": "/scratch/hs9g19/Mol_files_with_Hs/Mol_files_with_Hs",
        "opt": True,
        "coord": True,
        "all_dist": False,
        "second_neighbor": False,
    }

    train_dataset = open_dataset(storage_folder, front, "train", label_dict, args_dict)
    val_dataset = open_dataset(storage_folder, front, "val", label_dict, args_dict)

    y_values = [data.y for data in train_dataset]
    y_values = np.array(y_values)
    # print(data.y)
    classes = np.unique(y_values)
    weight = None

    _count_open_files()

    num_classes = len(label_dict)

    mp.spawn(
        main,
        args=(
            world_size,
            total_epochs,
            batch_size,
            train_dataset,
            val_dataset,
            best_path,
            ckp_path,
            num_classes,
            weight,
            classic_dropout,
            dropnode,
            mp_layers,
            edge_layers,
            fcc_in_gnn,
        ),
        nprocs=world_size,
    )

    _count_open_files()
