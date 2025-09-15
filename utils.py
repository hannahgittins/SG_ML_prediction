"""Hannah Gittins
utils for the RF model
"""

### importing libraries
import heapq
import operator
import os
import warnings

import numpy as np
import pandas as pd


def _check_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "level_0", "index"]
    df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        axis=1,
        inplace=True,
    )
    return df


def multi_precision_recall_accuracy(
    y_pred: list, y_test: list, classes: list, N: int
) -> tuple(float, pd.DataFrame):
    """Used to calculate the precision and recall for the topN space groups.

    Parameters
    ----------
    y_pred : list
        2D list of the predicted values with predictions associated with each space
    y_test : list
        list of the true values
    classes : list
        list of the space groups in the order they appear in y_pred
    N : int
        topN you are looking at

    Returns
    -------
    accuracy: float
        the accuracy of the model for the topN
    precision_recall_df: dataframe
        dataframe containing the precision and recall for each space group

    """
    ### creating dictionarys to contain the values
    true_positives = dict()
    false_negatives = dict()
    false_positives = dict()

    for target in classes:
        true_positives[target] = 0
        false_positives[target] = 0
        false_negatives[target] = 0

    accuracy_count = 0

    recall = list()
    precision = list()

    for i, pred in enumerate(y_pred):
        idx = list(
            zip(*heapq.nlargest(N, enumerate(pred), key=operator.itemgetter(1)))
        )[0]

        N_predicted = [classes[j] for j in idx]

        actual = y_test[i]

        if (
            actual in N_predicted
        ):  # if the space group that is assigned is correct then tp += 1 (if 1 is
            # correct in N - the space group has been assigned correctly and we ignore
            # the rest)
            accuracy_count += 1
            true_positives[actual] = true_positives[actual] + 1

        elif (
            actual not in N_predicted
        ):  # if the "actual" space group is not in the top N then we assign a false
            # negative and fp to every space group that has been falsey assigned to this
            # example
            try:
                false_negatives[actual] = false_negatives[actual] + 1
            except:
                print(actual)
                print(classes)

            for space_group in N_predicted:
                false_positives[space_group] = (
                    false_positives[space_group] + 1 / N
                )  # 1/N to scale and normalise because it is assigned N times for each
                # example in the top N block

    for target in classes:
        tp = true_positives[target]
        fp = false_positives[target]
        fn = false_negatives[target]

        try:
            recall.append((tp / (tp + fn)))
        except:
            recall.append(
                0
            )  # should be nan but for future functions to work we can assume 0

        try:
            precision.append((tp / (tp + fp)))
        except:
            precision.append(
                0
            )  # should be nan but for future fucntions to work we can assume 0

    precision = np.around(np.multiply(np.array(precision), 100), decimals=2)
    recall = np.around(np.multiply(np.array(recall), 100), decimals=2)

    accuracy = accuracy_count / len(y_pred)

    precision_recall_df = pd.DataFrame(
        {
            "space group": classes,
            f"precision top{N}": precision,
            f"recall top{N}": recall,
        }
    )

    precision_recall_df = precision_recall_df.set_index("space group")

    precision_recall_df = precision_recall_df.sort_values(
        by="space group", axis="index"
    )

    return accuracy, precision_recall_df


def prediction_frequency(y_pred: list, classes: list, N: int) -> dict:
    """Used to calculate the prediciton frequency for the top N predictions.

    Parameters
    ----------
    y_pred : list
        2D list of the predicted values with predictions associated with each space
        group
    classes : list
        ordering of the space groups within the array
    N : int
        topN you are looking at

    Returns
    -------
    pred_freq: dict
        dictionary containing the prediction frequency

    """
    pred_freq = {target: 0 for target in classes}

    for pred in y_pred:
        idx = list(
            zip(*heapq.nlargest(N, enumerate(pred), key=operator.itemgetter(1)))
        )[0]
        # returns the n top valus in the list, enumerate returns the idx and the value,
        # so operator.itemgetter(1) compares value, and then a tuple is returned and we
        # keep the index using [0]

        N_predicted = [classes[j] for j in idx]

        for N_target in N_predicted:
            pred_freq[N_target] = pred_freq[N_target] + 1

    return pred_freq


def multi_accuracy(y_pred: list, y_test: list, classes: list, N: int) -> float:
    """Used to calculate the accuracy for the topN space groups.

    Parameters
    ----------
    y_pred : list
        2D list of the predicted values with predictions associated with each space
        group
    y_test : list
        list of the true values
    classes : list
        list of the space groups in the order they appear in y_pred
    N : int
        topN you are looking at

    Returns
    -------
    accuracy: float
        the accuracy of the model for the topN

    """
    accuracy_count = 0

    for i, pred in enumerate(y_pred):
        idx = list(
            zip(*heapq.nlargest(N, enumerate(pred), key=operator.itemgetter(1)))
        )[0]
        # returns the n top valus in the list, enumerate returns the idx and the value,
        # so operator.itemgetter(1) compares value, and then a tuple is returned and we
        # keep the index using [0]

        N_predicted = [classes[j] for j in idx]

        actual = y_test[i]

        if (
            actual in N_predicted
        ):  # if the space group that is assigned is correct then tp += 1 (if 1 is
            # correct in N - the space group has been assigned correctly and we ignore
            # the rest)
            accuracy_count += 1

    accuracy = accuracy_count / len(y_pred)

    return accuracy


def _read_feature_file(filepath: str) -> list:
    with open(filepath) as f:
        features = f.read().splitlines()
        return [feat for feat in features if feat]


def RF_set_up_for_CV(
    final_folder: str,
    useless_feat_folder: str,
    to_include: dict,
    type_model: str,
    to_remove: list,
    geo_only_feat: list,
    chem_only_feat: list,
) -> tuple(str, list):
    """Set up the outpur folder names for the RF.

    Parameters
    ----------
    final_folder : str
        the folder filapath to place the results
    useless_feat_folder : str
        folder for the useless feats destination
    to_include : dict
        what features are being included in the training
    type_model : str
        what type of model is being trained (balanced or unbalanced)
    final_folder : str
        prefix for the final destination (if the data is balanced or unbalanced)
    to_remove : list
        what needs to be removed before training the model
    geo_only_feat : list
        features that need to removed to train a geo only model
    chem_only_feat : list
        features that need to removed to train a chem only model

    Returns
    -------
    accuracy_filename : str
        filepath to the final output destination
    to_remove : list
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
        accuracy_filename = os.path.join(
            folder, f"CV_{type_model}_accuracy{suffix}.csv"
        )
        return accuracy_filename

    if not geo_only and not useless_features_removed and not chem_only:
        accuracy_filename = set_file_paths(final_folder, "")

    elif geo_only and not useless_features_removed and not chem_only:
        accuracy_filename = set_file_paths(final_folder, "_geo_only")
        to_remove += geo_only_feat

    elif not geo_only and useless_features_removed and not chem_only:
        accuracy_filename = set_file_paths(
            final_folder, "_useless_features_removed_0_9"
        )
        to_remove += low_var_feat + high_corr_feat

    elif geo_only and useless_features_removed and not chem_only:
        accuracy_filename = set_file_paths(
            final_folder, "_geo_only_&_useless_features_removed_0_9"
        )
        to_remove += low_var_feat + high_corr_feat
        for chem in geo_only_feat:
            if chem not in to_remove:
                to_remove.append(chem)

    elif not geo_only and chem_only and not useless_features_removed:
        accuracy_filename = set_file_paths(final_folder, "_chem_only")
        to_remove += chem_only_feat

    elif not geo_only and chem_only and useless_features_removed:
        accuracy_filename = set_file_paths(
            final_folder, "_chem_only_&_useless_features_removed_0_9"
        )
        to_remove += low_var_feat + high_corr_feat
        for geo in chem_only_feat:
            if geo not in to_remove:
                to_remove.append(geo)

    return accuracy_filename, to_remove


def train_model(train: object, test: object, hyperparameters: dict):
    """Training the RF model.

    Parameters
    ----------
    train : dataframe
        the training set
    test : dataframe
        the test set
    hyperparameters : dict
        hyperparemeters for the model

    Returns
    -------
    top_N_accuracy : list
        list of the accuracies for top 1, 3, 5 and 10

    """
    from sklearn.ensemble import RandomForestClassifier

    train_y = train["target"].to_list()
    train_x = train.drop(columns=["target", "name"])
    test_y = test["target"].to_list()
    test_x = test.drop(columns=["target", "name"])

    model = RandomForestClassifier(
        n_estimators=hyperparameters["num_trees"],
        max_depth=hyperparameters["max_depth"],
        min_samples_split=hyperparameters["min_split"],
        min_samples_leaf=hyperparameters["min_leaf"],
        max_features=hyperparameters["max_features"],
        bootstrap=hyperparameters["bootstrap"],
    )

    model.fit(train_x, train_y)

    y_pred_prob = model.predict_proba(test_x)
    classes = model.classes_

    top_N_accuracy = []

    for i in [1, 3, 5, 10]:
        top_N_accuracy.append(multi_accuracy(y_pred_prob, test_y, classes, N=i))

    return top_N_accuracy
