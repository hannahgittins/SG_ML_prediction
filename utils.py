"""Utils used for the training of a GNN model."""

import os
from datetime import datetime

import pandas as pd
import psutil
import torch
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from graph_dataset import XYZDataset


def multi_precision_recall_accuracy(
    out: torch.Tensor, y_test: list, label_dict: dict, N: int
) -> tuple[float, pd.DataFrame]:
    """Used to calculate the precision and recall for the topN space groups.

    Parameters
    ----------
    out : torch.Tensor
        the output returned from the trained model
    y_test : list
        of the true values
    label_dict: dict
        dictionary containing the true value target (space group) and the given label
    N: int
        topN you are looking at

    Returns
    -------
    accuracy : float
        the accuracy of the model for the topN
    precision_recall_df : pd.DataFrame
        precision and recall stored in a pandas df

    """
    num_samples = out.size(0)
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    ### creating dictionarys to contain the values
    true_positives = {target: 0 for target in label_dict.keys()}
    false_positives = {target: 0 for target in label_dict.keys()}
    false_negatives = {target: 0 for target in label_dict.keys()}

    prob = torch.nn.functional.softmax(out, dim=1)
    _, N_pred = prob.topk(N, dim=1)
    N_pred = N_pred.tolist()

    classes = list(label_dict.keys())

    accuracy_count = 0

    for i in range(num_samples):
        N_predicted = [reverse_label_dict[pred] for pred in N_pred[i]]
        actual = reverse_label_dict[y_test.tolist()[i]]

        if actual in N_predicted:  # if the space group that is assigned is correct then
            # tp += 1 (if 1 is correct in N - the space group has
            # been assigned correctly and we ignore the rest)
            accuracy_count += 1
            true_positives[actual] += 1

        elif actual not in N_predicted:  # if the "actual" space group is not in the top
            # N then we assign a false negative and fp to
            # every space group that has been falsey
            # assigned to this example
            try:
                false_negatives[actual] += 1
            except:
                pass
                # print(actual)
                # print(classes)

            for space_group in N_predicted:
                false_positives[space_group] += 1 / N  # 1/N to scale and normalise
                # because it is assigned N times
                # for each example in the top N
                # block

    recall = []
    precision = []
    for target in classes:
        tp = true_positives[target]
        fp = false_positives[target]
        fn = false_negatives[target]

        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0

        recall.append(recall_val)
        precision.append(precision_val)

    accuracy = accuracy_count / num_samples

    precision_recall_df = (
        pd.DataFrame(
            {
                "space group": list(label_dict.keys()),
                f"precision top{N}": precision,
                f"recall top{N}": recall,
            }
        )
        .set_index("space group")
        .sort_index()
    )

    return accuracy, precision_recall_df


def multi_accuracy(
    out: torch.Tensor, y_test: torch.Tensor, label_dict: dict, N: int
) -> float:
    """Used to calculate the accuracy for the topN space groups

    Parameters
    ----------
    out: torch.Tensor
        output of the GNN prediciton
    y_test: torch.Tensor
        real targets
    label_dict: dict
        dictionary containing the true value target (space group) and the given label
    N: int
        top N

    Return
    ------
    accuracy: float
        accuracy for top N

    """
    num_samples = out.size(0)
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    prob = torch.nn.functional.softmax(out, dim=1)

    _, top_n_preds = prob.topk(N, dim=1)
    top_n_preds = top_n_preds.tolist()

    accuracy_count = 0

    for i in range(num_samples):
        N_predicted = [reverse_label_dict[pred] for pred in top_n_preds[i]]
        actual = reverse_label_dict[y_test.tolist()[i]]

        if actual in N_predicted:  # if the space group that is assigned is correct then
            # tp += 1 (if 1 is correct in N - the space group has
            # been assigned correctly and we ignore the rest)
            accuracy_count += 1

    accuracy = accuracy_count / num_samples

    return accuracy


class SaveBest:
    """Object that is used to update the model that is saved if a model with a different
    number of epochs performs better than the previous.
    """

    def __init__(self, best_loss: float = float("inf"), verbose: bool = True) -> None:
        """Initialise  object.

        Parameters
        ----------
        best_loss: float
            the best loss value
        verbose: bool

        Return
        ------
        None

        """
        self.best_loss = best_loss
        self.verbose = verbose

    def __call__(
        self,
        current_loss: float,
        epoch: int,
        model: object,
        optimizer: object,
        criterion: object,
        path: str,
    ) -> None:
        """Save the best model.

        Parameters
        ----------
        current_loss : float
            validation loss of best model
        epoch : int
            the epoch at which this model was saved
        model : object
            the model at the current state
        optimizer : object
            the optimizer at the current state
        criterion : object
            the criterion at the current state
        path : str
            filepath of the model to be saved at

        Returns
        -------
        None -> saves the model at the filepath

        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            print(
                f"Saving new model with loss of {current_loss} after {epoch} epoch(s)"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                    "losslogger": current_loss,
                },
                path,
            )


def load_checkpoint(model: object, optimizer: object, filepath: str):
    """Loads in previous checkpoint.

    Parameters
    ----------
    model : object
        graph neural network model
    optimizer : object
        pytorch optimizer
    filepath : str
        filepath to the last checkpoint

    Returns
    -------
    model : object
        model from the checkpoint
    optimizer : object
        the optimizer as the sate of this checkpoint
    start_epoch : int
        the epoch that the model was left at
    loss : float
        the loss of the model at this checkpoint

    """
    print(f"Loading in last checkpoint: {filepath}")
    checkpoint = torch.load(
        filepath, weights_only=False
    )  ## added weights_only=True recently!
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["losslogger"]
    print(
        f"Check point loaded => Current validation loss: {loss} at epoch: {start_epoch}"
    )

    return model, optimizer, start_epoch, loss


def create_dataset_with_smiles(
    smiles: list, labels: list, label_dict: dict = None
) -> tuple[list, dict]:
    """Creates a dataset of molecules on the fly using the smile strings.

    Parameters
    ----------
    smiles : list of strings
        list of smiles strings
    labels : list of ints
        of the labels for each of the smiles
    label_dict : dict
        label to token relationship (default None)

    Return
    ------
    dataset : list
        of the molecular graphs containing y values
    label_dict : dict
        label to token relationship

    """
    from molecular_graph_creation import molecular_graph_with_smiles

    def _gen_list():
        for smile, label in zip(smiles, labels):
            yield molecular_graph_with_smiles(smile, label, label_dict)

    if label_dict is None:
        label_dict = {label: i for i, label in enumerate(set(labels))}

    return list(_gen_list()), label_dict


def create_dataset_with_xyz_files(
    refcodes: list,
    xyz_folder: str,
    mol_folder: str,
    labels: list,
    label_dict: dict = None,
):
    """Creates a dataset of molecules on the fly using the smile strings.

    Paramaters
    ----------
    refcodes: list
        list of refcodes
    xyz_folder: str
        filepath to the xyz files
    mol_folder: str
        filepath to the mol files (if necessary)
    labels: list
        target associated with refcode
    label_dict:
        label to token relationship

    Returns
    -------
    dataset: list
        the list of the molecular graphs containing y values
    label_dict: dict
        label to token relationship

    """
    from molecular_graph_creation import molecular_graph_with_xyz

    def _gen_list():
        for refcode, label in zip(refcodes, labels):
            yield molecular_graph_with_xyz(
                refcode, xyz_folder, mol_folder, label, label_dict, coord=True, opt=True
            )

    if label_dict is None:
        label_dict = {label: i for i, label in enumerate(set(labels))}

    return list(_gen_list()), label_dict


def log_memory_usage(note: str = "", log_file: str = "memory_log.txt") -> None:
    """Log memory used.

    Parameters
    ----------
    note: str
        note with memeory log
    log_file: str
        file name

    Return
    ------
    None

    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e9  # Memory in GB
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a") as f:
        f.write(f"[{timestamp}] [{note}] Memory usage: {mem:.2f} GB\n")


def open_dataset(
    storage_folder: str,
    front: str,
    dataset_name: str,
    label_dict: dict,
    args_dict: dict,
) -> object:
    """Open dataset ready for training.

    Parameters
    ----------
    storage_folder: str
        folder containing the molecular graphs
    front: str
        front half of the file
    dataset_name: str
        end half of file name and the dataset type
    label_dict: dict
        tokenized labels dict
    args_dict: dict
        arguments for the stored molecular graphs

    Return
    ------
    molecular_graphs: object
        returns an torch object containing the molecular graphs ready for training

    """
    root = os.path.join(f"{storage_folder}", f"{front}_{dataset_name}")
    process = os.path.join(root, "processed")

    file_names = [
        os.fsdecode(file) for file in os.listdir(process) if "data" in os.fsdecode(file)
    ]
    if len(file_names) == 0:
        file_names = None

    return XYZDataset(
        root=root,
        raw_file=f"{dataset_name}.csv",
        file_names=file_names,
        args_dict=args_dict,
        label_dict=label_dict,
    )


def setup(local_rank: int, world_size: int, seed: int = 37) -> None:
    """Setting up the GPUs for ddp.

    Parameters
    ----------
    local_rank: int
        the gpu label
    world_size: int
        number of gpus
    seed: int
        random seed

    Returns
    -------
    None

    """
    ### defining the seed:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"

    torch.cuda.set_device(local_rank)
    init_process_group(backend="nccl", rank=local_rank, world_size=world_size)


def prepare_dataloader(dataset: object, batch_size: int, shuffle: bool = False):
    """Prepare dataloader.

    Parameters
    ----------
    dataset: object
        dataset to prep
    batch_size: int
        batch size
    shuffle: bool
        default=False, shuffle dataset

    Return
    ------
    dataloader: oject
        torch dataloader raedy for training

    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        sampler=DistributedSampler(dataset),
    )


def _count_open_files():
    pid = os.getpid()
    num_files = len(os.listdir(f"/proc/{pid}/fd"))
    print(f"Open files: {num_files}")
    return num_files
