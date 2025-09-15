"""Script used to run a random forest and caculate the accuracy of the first, top3 and
top5 space groups
"""

### importing libraries
import os
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from utils import (
    _check_and_clean,
    multi_precision_recall_accuracy,
    prediction_frequency,
)


def _read_feature_file(filepath):
    with open(filepath) as f:
        features = f.read().splitlines()
    return [feat for feat in features if feat]


def RF_set_up(
    final_folder: str,
    useless_feat_folder: str,
    to_include: dict,
    type_model: str,
    to_remove: list,
    geo_only_feat: list,
    chem_only_feat: list,
):
    """Set up the output folder names for the RF.

    Parameters
    ----------
    final_folder: str
        the folder filepath to place the results
    useless_feat_folder: str
        folder for the useless feats dest
    to_include: dict
        what features are being included in the training
    type_model: str
        prefix for the final destination (if the data is balanced or unabalanced)
    geo_only_feat: list
        features that need to removed to train a geo only model
    chem_only_feat : list
        features that need to removed to train a chem only model

    Return
    ------
    [pkl_filename:str, accuracy_filename:str, precision_filename:str,
     prediction_filename:str]
        filepaths to the final output desitination
    to_remove: list
        the list of features to remove

    """
    geo_only, useless_features_removed, chem_only = list(to_include.values())

    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    if geo_only == True and chem_only == True:
        warnings.warm(
            "You can't have geo only and chem only - this will proceed with geo only."
        )

    if useless_features_removed:
        high_corr_feat = _read_feature_file(
            os.path.join(
                useless_feat_folder, "RF_all_organics_high_corr_features_0_9.txt"
            )
        )
        low_var_feat = _read_feature_file(
            os.path.join(
                useless_feat_folder, "RF_all_organics_low_variance_features.txt"
            )
        )

    def set_file_paths(folder: str, suffix: str):
        pkl_filename = os.path.join(folder, f"RF_{type_model}_model{suffix}.joblib")
        accuracy_filename = os.path.join(
            folder, f"RF_{type_model}_accuracy{suffix}.txt"
        )
        precision_filename = os.path.join(
            folder, f"RF_{type_model}_precision_recall{suffix}.csv"
        )
        prediction_filename = os.path.join(
            folder, f"RF_{type_model}_prediction_frequency{suffix}.csv"
        )
        return pkl_filename, accuracy_filename, precision_filename, prediction_filename

    if not geo_only and not useless_features_removed and not chem_only:
        pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
            set_file_paths(final_folder, "")
        )
    elif geo_only and not useless_features_removed and not chem_only:
        pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
            set_file_paths(final_folder, "_geo_only")
        )
        to_remove += geo_only_feat
    elif not geo_only and useless_features_removed and not chem_only:
        pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
            set_file_paths(final_folder, "_useless_features_removed_0_9")
        )
        to_remove += low_var_feat + high_corr_feat
    elif geo_only and useless_features_removed and not chem_only:
        pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
            set_file_paths(final_folder, "_geo_only_&_useless_features_removed_0_9")
        )
        to_remove += low_var_feat + high_corr_feat
        for chem in geo_only_feat:
            if chem not in to_remove:
                to_remove.append(chem)
    elif not geo_only and chem_only and not useless_features_removed:
        pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
            set_file_paths(final_folder, "_chem_only")
        )
        to_remove += chem_only_feat
    elif not geo_only and chem_only and useless_features_removed:
        pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
            set_file_paths(final_folder, "_chem_only_&_useless_features_removed_0_9")
        )
        to_remove += low_var_feat + high_corr_feat
        for geo in chem_only_feat:
            if geo not in to_remove:
                to_remove.append(geo)

    return [
        pkl_filename,
        accuracy_filename,
        precision_filename,
        prediction_filename,
    ], to_remove


def RF_train_model(
    train_path: str,
    test_path: str,
    final_filepaths: list,
    hyperparameters: dict,
    to_remove: list,
) -> None:
    """Training the RF model with the results outputted to the final destination.

    Parameters
    ----------
    train_path: str
        the training set filapath
    test_path: str
        the test set filepath
    final_filepaths: list
        the final dest filepaths
    hyperparameters: dict
        the hyperparameters for training the model
    to_remove: list
        features to remove

    Returns
    -------
    None

    """
    pkl_filename, accuracy_filename, precision_filename, prediction_filename = (
        final_filepaths
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = _check_and_clean(train)
    test = _check_and_clean(test)

    y_train = train["target"].to_list()
    X_train = train.drop(to_remove, axis=1)

    y_test = test["target"].to_list()
    X_test = test.drop(to_remove, axis=1)

    num_trees, min_split, min_leaf, max_features, max_depth, bootstrap = list(
        hyperparameters.values()
    )

    model = RandomForestClassifier(
        n_estimators=num_trees,
        max_depth=max_depth,
        min_samples_split=min_split,
        min_samples_leaf=min_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
    )

    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)
    classes = model.classes_

    dump(model, pkl_filename, compress=3)

    acc_dict = {}
    prec_recall_dict = {}
    pred_freq_dict = {}

    with open(accuracy_filename, "w") as f:
        for i in [1, 3, 5, 10]:
            acc_dict[i], prec_recall_dict[i] = multi_precision_recall_accuracy(
                y_pred_prob, y_test, classes, i
            )
            f.write(f"Top {i} accuracy: {round((100 * acc_dict[i]), 2)}\n")

            temp_pred_freq = prediction_frequency(y_pred_prob, classes, i)
            temp_pred_list = list(temp_pred_freq.values())
            denominator = sum(temp_pred_list) / i
            pred_freq_dict[i] = np.around(
                np.multiply(np.divide(np.array(temp_pred_list), denominator), 100),
                decimals=2,
            )

    data = pd.concat(prec_recall_dict, axis=1)
    data.to_csv(precision_filename)

    pred_frequency_dataframe = pd.DataFrame(
        {
            "Space group": classes,
            "Prediction Frequency top 1": pred_freq_dict[1],
            "Prediction Frequency top 3": pred_freq_dict[3],
            "Prediction Frequency top 5": pred_freq_dict[5],
            "Prediction Frequency top 10": pred_freq_dict[10],
        }
    )
    pred_frequency_dataframe.to_csv(prediction_filename)


def train(
    train_file: str,
    test_file: str,
    useless_feat_folder: str,
    final_folder: str,
    dataset_name: str,
    to_include: dict,
    hyperparamters: dict,
) -> None:
    """Train RF model.

    Parameters
    ----------
    train_file: str
        training csv filepath
    test_file: str
        test csv filepath
    useless_feat_folder: str
        useless features folder
    final_folder: str
        output folder
    dataset_name: str
        dataset name e.g. "balanced"
    to_include : dict
        dictionary containing booleans for whether to include geo, use and chem features
    hyperparamters : dict
        dictionary containing the hyperparamters for the RF

    Return
    ------
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

    final_filepaths, to_remove = RF_set_up(
        final_folder,
        useless_feat_folder,
        to_include,
        dataset_name,
        to_remove,
        geo_only_feat,
        chem_only_feat,
    )

    RF_train_model(train_file, test_file, final_filepaths, hyperparamters, to_remove)


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

    train(
        train_file,
        test_file,
        useless_feat_folder,
        final_folder,
        dataset_name,
        to_include,
        hyperparamters,
    )
