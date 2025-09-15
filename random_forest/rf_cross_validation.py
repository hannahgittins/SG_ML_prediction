"""Hannah Gittins
Cross validation over the unbalanced dataset
"""

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from utils import RF_set_up_for_CV, _check_and_clean, train_model


def _bal_calculate_random_accuracy(Y_test: list, top_N: int) -> float:
    counter = 0
    N = len(Y_test)
    target_keys = list(set(Y_test))
    num_targets = len(target_keys)

    if top_N == 1:
        random_idx = np.random.choice(num_targets, size=N)
        top_1_pred_targets = [target_keys[idx] for idx in random_idx]

        for i, actual in enumerate(Y_test):
            if actual == top_1_pred_targets[i]:
                counter += 1

    elif top_N != 1:
        top_N_matrix = np.zeros(shape=(N, top_N))

        for i in range(N):
            random_idx = np.random.choice(num_targets, size=top_N, replace=False)
            top_i_pred_targets = [target_keys[idx] for idx in random_idx]
            top_N_matrix[i, :] = top_i_pred_targets

        for i, actual in enumerate(Y_test):
            if actual in top_N_matrix[i, :]:
                counter += 1

    return counter / N


def bal_calculate_random_accuracies(Y_test: list, iterations: int) -> list:
    """Calculate the accuracies over n interations for the top 1, 3, 5, and 10 targets.

    Parameters
    ----------
    Y_test: list
        the acutal target values
    iterations: list
        number of iteration to run over to get a avarage random result

    Returns
    -------
    accuracies: list
        "accuracies" for the top 1, 3, 5 and 10 space groups

    """
    accuracies = {}

    for i in range(iterations):
        for n in [1, 3, 5, 10]:
            rand_acc = _bal_calculate_random_accuracy(Y_test, n)
            if i == 0:
                accuracies[n] = [rand_acc]
            else:
                accuracies[n].append(rand_acc)

    for n in [1, 3, 5, 10]:
        accuracies[n] = np.mean(accuracies[n])

    return accuracies


def bal_data_splitting(data: pd.DataFrame, CV: int, even_cvs: bool) -> tuple(
    dict, dict
):
    """Splitting the data into N CV for CV.

    Parameters
    ----------
    data: object
        the whole dataset for CV
    CV: int
        the number of cross validaitons for CV
    even_cvs: bool
        true if want all the cv datasets to be the same size

    Returns
    -------
    test_name_dict: dict
        dictionary containing lists of names for the N CVs
    train_name_dict: dict
        dictionary containing lists of names for the N CVs

    """
    target = list(set(data["target"].to_list()))

    target_name_dict = dict()
    for t in target:
        temp = data[data["target"] == t]["name"].to_list()
        random.shuffle(temp)
        target_name_dict[t] = temp

    test_name_dict = dict()
    train_name_dict = dict()

    steps = len(temp) // CV

    for cv in range(CV):
        train = []
        test = []
        for t in target:
            temp = list(target_name_dict[t])
            if cv == (CV - 1) and even_cvs == False:
                test_temp = temp[cv * steps :]
            else:
                test_temp = temp[cv * steps : (cv + 1) * steps]

            train_temp = list(set(temp) - set(test_temp))

            train += train_temp
            test += test_temp

        test_name_dict[cv + 1] = test
        train_name_dict[cv + 1] = train

    return test_name_dict, train_name_dict


def balanced_cross_validation(
    og_train: pd.DataFrame,
    og_test: pd.DataFrame,
    hyperparameter: dict,
    CV: int = 5,
    even_cvs: bool = False,
    iterations: int = 100,
) -> tuple(dict, dict, dict, dict):
    """Cross validation over the balanced dataset (maintianing the balanced space groups
    within the test and training set).

    Parameters
    ----------
    og_train: object
        the original training set
    og_test: object
        the original test set
    hyperparameter: dict
        dictionary containing the hyperparamters for the RF
    CV: int
        the number of cross validations (max CV for less than 10000 = 10 and max overall
        is 15)
    even_cvs: bool
        true if want all the cv datasets to be the same size
    iterations: int
        number of iterations for the random accuracy calculation (default = 100)

    Returns
    -------
    final_accuracy: dict
        the mean accuracy for the cross validation steps
    accuracy_std: dict
        the std for the cross validation steps
    final_rand_accuracy: dict
        the mean random accuracy for the cross validation steps
    rand_accuracy_std: dict
        the std of the random accuracy for the cross validation steps

    """
    data = pd.concat([og_train, og_test])

    if CV > 10 and len(data) < 10000:
        raise (
            "Can't have CV greater then 10 when the dataset is smaller than 10000 molecules"
        )
    if CV > 15:
        raise ("Cut off for CV is 15")

    test_name_dict, train_name_dict = bal_data_splitting(data, CV, even_cvs)

    accuracy_dict = {}
    rand_accuracy_dict = {}

    for cv in range(1, CV + 1):
        test_names = test_name_dict[cv]
        train_names = train_name_dict[cv]

        test_data = data[data["name"].isin(test_names)]
        train_data = data[data["name"].isin(train_names)]

        top_01_accuracy, top_03_accuracy, top_05_accuracy, top_10_accuracy = (
            train_model(train_data, test_data, hyperparameter)
        )
        rand_accuracy = bal_calculate_random_accuracies(
            test_data["target"].to_list(), iterations=iterations
        )

        for n, acc in zip(
            [1, 3, 5, 10],
            [top_01_accuracy, top_03_accuracy, top_05_accuracy, top_10_accuracy],
        ):
            rand_acc = rand_accuracy[n]
            if cv == 1:
                accuracy_dict[n] = [acc]
                rand_accuracy_dict[n] = [rand_acc]
            else:
                accuracy_dict[n].append(acc)
                rand_accuracy_dict[n].append(rand_acc)

    final_accuracy = {}
    accuracy_std = {}
    final_rand_accuracy = {}
    rand_accuracy_std = {}

    for n in [1, 3, 5, 10]:
        acc = accuracy_dict[n]
        final_accuracy[n] = round(np.mean(acc) * 100, 2)
        accuracy_std[n] = round(np.std(acc) * 100, 2)

        rand_acc = rand_accuracy_dict[n]
        final_rand_accuracy[n] = round(np.mean(rand_acc) * 100, 2)
        rand_accuracy_std[n] = round(np.std(rand_acc) * 100, 2)

    return final_accuracy, accuracy_std, final_rand_accuracy, rand_accuracy_std


def _unbal_calculate_random_accuracy(y_test_dict: dict, labels: any, N: int) -> float:
    if type(labels) != list:
        values = y_test_dict[labels]
        accuracy = np.sum(values) / N
    else:
        values = [y_test_dict[label] for label in labels]
        accuracy = np.sum(values) / N

    return accuracy


def unbal_calculate_random_accuracies(target: list) -> float:
    """Calculate the random accuracies for the top 1, 3, 5 and 10 space groups for the
    unbalanced dataset.

    Parameters
    ----------
    target : list
        list of the target space groups in the test set

    Return
    ------
    accuracies : dict
        dictionary containing the random accuracies for the top 1, 3, 5 and 10 space
        groups

    """
    N = len(target)
    target_dict = Counter(target)
    target_dict = dict(sorted(target_dict.items(), key=lambda item: item[1]))

    top_1 = 14
    top_3s = [14, 19, 2]
    top_5s = [14, 19, 2, 4, 61]
    top_10s = [14, 19, 2, 4, 61, 15, 33, 9, 29, 5]

    accuracies = {}

    for i, top_i in zip([1, 3, 5, 10], [top_1, top_3s, top_5s, top_10s]):
        accuracies[i] = _unbal_calculate_random_accuracy(target_dict, top_i, N)

    return accuracies


def unbal_data_splitting(data: pd.DataFrame, CV: int, even_cvs: bool):
    """Splitting the data into N CV for CV for unbalanced.

    Parameters
    ----------
    data : dataframe
        the whole dataset for CV
    CV : int
        the number of cross validaitons for CV
    even_cvs : bool
        true if want all the cv datasets to be the same size

    Returns
    -------
    test_name_dict : dict
        dictionary containing lists of names for the N CVs
    train_name_dict : dict
        dictionary containing lists of names for the N CVs

    """
    names = data["name"].to_list()
    random.shuffle(names)

    steps = len(names) // CV

    test_name_dict = {}
    train_name_dict = {}

    for cv in range(CV):
        if cv == (CV - 1) and even_cvs == False:
            test_temp = names[cv * steps :]
        else:
            test_temp = names[cv * steps : (cv + 1) * steps]

        train_temp = list(set(names) - set(test_temp))

        test_name_dict[cv + 1] = test_temp
        train_name_dict[cv + 1] = train_temp

    return test_name_dict, train_name_dict


def unbalanced_cross_validation(
    og_train: pd.DataFrame,
    og_test: pd.DataFrame,
    hyperparameter: dict,
    CV: int = 5,
    even_cvs: bool = False,
) -> dict:
    """Cross validation over the balanced dataset (maintianing the balanced space groups
    within the test and training set).

    Parameters
    ----------
    og_train : dataframe
        the original training set
    og_test : dataframe
        the original test set
    hyperparameter : dict
        dictionary containing the hyperparamters for the RF
    CV : int, optional
        the number of cross validations (max CV for less than 10000 = 10 and max overall
        is 15), by default 5
    even_cvs : bool, optional
        true if want all the cv datasets to be the same size, by default False

    Returns
    -------
    final_accuracy : dict
        the mean accuracy for the cross validation steps
    accuracy_std : dict
        the standard deviation of the accuracies for the cross validation steps
    final_rand_accuracy : dict
        the mean random accuracy for the cross validation steps
    rand_accuracy_std : dict
        the standard deviation of the random accuracies for the cross validation steps

    """
    data = pd.concat([og_train, og_test])

    if CV > 10 and len(data) < 10000:
        raise (
            "Can't have CV greater then 10 when the dataset is smaller than 10000 molecules"
        )
    if CV > 15:
        raise ("Cut off for CV is 15")

    test_name_dict, train_name_dict = unbal_data_splitting(data, CV, even_cvs)

    accuracy_dict = {}
    random_accuracy_dict = {}

    for cv in range(1, CV + 1):
        test_names = test_name_dict[cv]
        train_names = train_name_dict[cv]

        test_data = data[data["name"].isin(test_names)]
        train_data = data[data["name"].isin(train_names)]

        top_01_accuracy, top_03_accuracy, top_05_accuracy, top_10_accuracy = (
            train_model(train_data, test_data, hyperparameter)
        )
        random_accuracy = unbal_calculate_random_accuracies(
            test_data["target"].to_list()
        )

        for n, acc in zip(
            [1, 3, 5, 10],
            [top_01_accuracy, top_03_accuracy, top_05_accuracy, top_10_accuracy],
        ):
            rand_acc = random_accuracy[n]
            if cv == 1:
                accuracy_dict[n] = [acc]
                random_accuracy_dict[n] = [rand_acc]
            else:
                accuracy_dict[n].append(acc)
                random_accuracy_dict[n].append(rand_acc)

    final_accuracy = {}
    accuracy_std = {}
    final_rand_accuracy = {}
    rand_accuracy_std = {}

    for n in [1, 3, 5, 10]:
        acc = accuracy_dict[n]
        final_accuracy[n] = round(np.mean(acc) * 100, 2)
        accuracy_std[n] = round(np.std(acc) * 100, 2)

        rand_acc = random_accuracy_dict[n]
        final_rand_accuracy[n] = round(np.mean(rand_acc) * 100, 2)
        rand_accuracy_std[n] = round(np.std(rand_acc) * 100, 2)

    return final_accuracy, accuracy_std, final_rand_accuracy, rand_accuracy_std


def cross_val(
    train_file: str,
    test_file: str,
    useless_feat_folder: str,
    final_folder: str,
    dataset_name: str,
    to_include: dict,
    hyperparamters: dict,
    CV: int,
) -> None:
    """Cross validation over the unbalanced dataset

    Parameters
    ----------
    train_file : str
        filepath to the training dataset
    test_file : str
        filepath to the test dataset
    useless_feat_folder : str
        filepath to the folder containing the useless feature files
    final_folder : str
        filepath to the folder where the results will be saved
    dataset_name : str
        name of the dataset
    to_include : dict
        dictionary containing booleans for whether to include geo, use and chem features
    hyperparamters : dict
        dictionary containing the hyperparamters for the RF
    CV : int
        the number of cross validations (max CV for less than 10000 = 10 and max overall
        is 15)

    Returns
    -------
    None

    """
    geo_only_feat = [
        "Charge",
        "Topological polar surface area",
        "Alcohol",
        "Amide",
        "Imine",
        "Nitro group",
        "Nitrile",
        "Amine",
        "Ether",
        "Aldehyde",
        "Halide",
        "Ketone",
        "Carboxylic acid",
        "Anhydride",
        "Ester",
        "Thiol",
        "Thiocarbonyl",
        "Thioether",
        "Sulfone",
        "Phosphoric acid",
    ]
    chem_only_feat = [
        "Aspherical",
        "Eccentricity",
        "Interial Shape Factor",
        "Normalised principle moment ratio 1",
        "Normalised principle moment ratio 2",
        "First (smallest) principle moment of inertia",
        "Second principle moment of inertia",
        "Third (largest) prinicple moment of inertia",
        "Radius of Gyration",
        "Molecule sphericity index",
        "Plane of best fit",
        "Molecular weight",
        "Number of atoms",
        "Number of rings",
        "Number of aromatic rings",
        "Number of operations",
        "E",
        "C2",
        "C2'",
        "C2''",
        "C2(z)",
        "C2(y)",
        "C2(x)",
        "C3",
        "C3^(2)",
        "C4",
        "C4^(3)",
        "C5",
        "C5^(2)",
        "C5^(3)",
        "C5^(4)",
        "C6",
        "C6^(5)",
        "C7",
        "C7^(2)",
        "C7^(3)",
        "C7^(4)",
        "C7^(5)",
        "C7^(6)",
        "C8",
        "C8^(3)",
        "C8^(5)",
        "C8^(7)",
        "?h",
        "?v",
        "?d",
        "?v(xz)",
        "?'v(yz)",
        "?(xy)",
        "?(xz)",
        "?(yz)",
        "i",
        "S3",
        "S3^(5)",
        "S4",
        "S4^(3)",
        "S5",
        "S5^(3)",
        "S5^(7)",
        "S5^(9)",
        "S6",
        "S6^(5)",
        "S8",
        "S8^(3)",
        "S8^(5)",
        "S8^(7)",
        "S10",
        "S10^(3)",
        "S12",
        "S12^(5)",
    ]

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    train_data = _check_and_clean(train_data)
    test_data = _check_and_clean(test_data)

    final_filepath, to_remove = RF_set_up_for_CV(
        final_folder,
        useless_feat_folder,
        to_include,
        dataset_name,
        to_remove,
        geo_only_feat,
        chem_only_feat,
    )

    train_i = train_data.drop(columns=to_remove, inplace=False)
    test_i = test_data.drop(columns=to_remove, inplace=False)

    if dataset_name.lower() == "balanced":
        final_accuracy, accuracy_std, final_rand_accuracy, rand_accuracy_std = (
            balanced_cross_validation(
                train_i, test_i, hyperparamters, CV=CV, even_cvs=False
            )
        )
    elif dataset_name.lower() == "unbalanced":
        final_accuracy, accuracy_std, final_rand_accuracy, rand_accuracy_std = (
            unbalanced_cross_validation(
                train_i, test_i, hyperparamters, CV=CV, even_cvs=False
            )
        )

    top_N = np.array(list(final_rand_accuracy.keys()))
    rand_acc = np.array(list(final_rand_accuracy.values()))
    rand_std = np.array(list(rand_accuracy_std.values()))
    acc = np.array(list(final_accuracy.values()))
    std = np.array(list(accuracy_std.values()))

    overlap_val = np.subtract(np.subtract(acc, std), np.add(rand_acc, rand_std))
    overlap = ["Y" if val < 0 else "N" for val in overlap_val]

    temp = {
        "Top N": top_N,
        "Random accuracy": rand_acc,
        "Rand std": rand_std,
        "Accuracy": acc,
        "Std": std,
        "Overlap?": overlap,
    }

    data = pd.DataFrame(temp)
    data.set_index("Top N", inplace=True)

    print(final_filepath)
    print(data)

    data.to_csv(final_filepath)


if __name__ == "__main__":
    train_file = None
    test_file = None
    useless_feat_folder = None
    final_folder = None
    dataset_name = None  # e.g. "unbalanced" or "balanced"
    to_include = {"geo": bool, "use": bool, "chem": bool}
    hyperparamters = {
        "num_trees": int,
        "min_split": int,
        "min_leaf": int,
        "max_features": str,
        "max_depth": int,
        "bootstrap": bool,
    }
    CV = None

    cross_val(
        train_file,
        test_file,
        useless_feat_folder,
        final_folder,
        dataset_name,
        to_include,
        hyperparamters,
        CV,
    )
