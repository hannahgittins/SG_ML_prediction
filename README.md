# SG_ML_prediction
Scripts necessary to repeat results (train models) from the Space group exploration study. 

## Python script breakdown:
gnn/EGNN_edit_model.py -> EGNN model used (editted to include FCC layers post message passing layers) [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)

gnn/EGNN_train.py -> training of the EGNN model 

gnn/test_egnn_model.py -> test EGNN model


gnn/MPNN_model.py -> original untouched MPNN model (NNConv) [arXiv:1704.01212](https://arxiv.org/abs/1704.01212), [arXiv:1704.02901](https://arxiv.org/abs/1704.02901)

gnn/MPNN_train.py -> train MPNN model

gnn/MPNN_train_augmentation.py -> train model with data augmentation

gnn/MPNN_train_no_coord.py -> train model with no coordinates

gnn/test_model.py -> test original and data augmentated MPNN model 

gnn/test_model_no_coord.py -> test model with no coordinates


gnn/utils.py -> contains functions used in training and test scripts

gnn/graph_dataset.py -> used to load in the molecualr graphs


random_forest/rf_model.py -> train random forest models

random_forest/rf_cross_validation.py -> random forest cross validation


random_forest/hypertuning_optuna.py -> random forest hyperparameter tuning

random_forest/utils.py -> functions present in other random forest scripts


## Dependancies
PyTorch [arXiv:1912.01703](https://arxiv.org/abs/1912.01703), 
PyTorch Geometric [arXiv:1903.02428](https://arxiv.org/abs/1903.02428), 
Scikit-Learn [arXiv:1201.0490](https://arxiv.org/abs/1201.0490), 
Optuna [arXiv:1907.10902](https://arxiv.org/abs/1907.10902),
RDKit (https://www.rdkit.org/)
