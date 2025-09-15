"""Script used to discover the best parameters for a given random forest"""

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def _check_and_clean(df):
    columns_to_drop = ["Unnamed: 0", "Unnamed: 0.1", "Unnamed: 0.2", "level_0", "index"]
    df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        axis=1,
        inplace=True,
    )
    return df


def hyperparameter_tuning(
    train_file: str,
    val_file: str,
    tree_limits: list = [10, 1000, 10],
    max_feat_limits: list = ["log2", "sqrt"],
    max_depth_limits: list = [10, 100, 10],
    node_limits: list = [2, 10, 1],
    leaf_limits: list = [1, 10, 1],
    boot_limits=[True, False],
):
    """Function to hypertune a random forest classifier using optuna.

    Parameters
    ----------
    train_file : str
        file path to the training dataframe
    val_file : str
        file path to the validation dataframe
    tree_limits : list
        list of three integers [min, max, step] for the number of trees to search
        between
    max_feat_limits : list
        list of strings of the max features to search between
    max_depth_limits : list
        list of three integers [min, max, step] for the max depth of the trees to search
        between
    node_limits : list
        list of three integers [min, max, step] for the min number of branches that
        split a node to search between
    leaf_limits : list
        list of three integers [min, max, step] for the min number of samples required
        per leaf to search between
    boot_limits : list
        list of booleans for whether to bootstrap or not

    Returns
    -------
    None
        prints the best parameters and their corresponding accuracy score on the
        validation set

    """
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)

    train = _check_and_clean(train)
    y_train = train["target"].to_list()
    X_train = train.drop(["SMILES", "target", "name"], axis=1)

    val = _check_and_clean(val)
    y_val = val["target"].to_list()
    X_val = val.drop(["SMILES", "target", "name"], axis=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    def _objective(
        trial,
        tree_limits: list = tree_limits,
        max_feat_limits: list = max_feat_limits,
        max_depth_limits: list = max_depth_limits,
        node_limits: list = node_limits,
        leaf_limits: list = leaf_limits,
        boot_limits: list = boot_limits,
    ):
        trees = trial.suggest_int(
            "number of trees", tree_limits[0], tree_limits[1], step=tree_limits[2]
        )
        max_feat = trial.suggest_categorical("max features", max_feat_limits)
        max_depth = trial.suggest_int(
            "max depth of the trees",
            max_depth_limits[0],
            max_depth_limits[1],
            step=max_depth_limits[2],
        )
        node_min_split = trial.suggest_int(
            "min number of branches that split a node",
            node_limits[0],
            node_limits[1],
            step=node_limits[2],
        )
        samples_leaf = trial.suggest_int(
            "min number of samples required per leaf",
            leaf_limits[0],
            leaf_limits[1],
            step=leaf_limits[2],
        )
        bootstrap = trial.suggest_categorical("boostraping or not", boot_limits)

        random_forest = RandomForestClassifier(
            n_estimators=trees,
            max_depth=max_depth,
            min_samples_split=node_min_split,
            min_samples_leaf=samples_leaf,
            max_features=max_feat,
            bootstrap=bootstrap,
        )
        random_forest.fit(X_train, y_train)

        val_pred = random_forest.predict(X_val)

        return metrics.accuracy_score(y_val, val_pred)


if __name__ == "__main__":
    hyperparameter_tuning(
        train_file=None,
        val_file=None,
    )
