"""Editted EGNN model."""

import torch
import torch_geometric
from egnn_pytorch.egnn_pytorch_geometric import EGNN_Sparse
from torch import nn
from torch_geometric import nn as gnn


class EGNN_edit(nn.Module):
    def __init__(
        self,
        example: object,
        num_classes: int,
        egnn_layers: int = 3,
        fc_layers: int = 2,
        hidden: int = 32,
    ) -> None:
        """Initalizing the model.

        Parameters
        ----------
        example: object
            molecular graph example
        num_classes : int
            number of the targets (therefore size of the output layer)
        egnn_layers : int
            dafault =3 , number of egnn layers
        fc_layers : int
            default =2, number of the fully connected layers after the message passing
        hidden : int
            default =32, the number of nodes in the hidden layers

        Return
        ------
        None

        """
        super(EGNN_edit, self).__init__()

        self.num_fc = fc_layers
        self.fcs = nn.ModuleList()
        self.egnn = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(egnn_layers):
            egnn = EGNN_Sparse(
                feats_dim=(example.num_node_features - 3),
                pos_dim=3,
                edge_attr_dim=(example.num_edge_features),
                soft_edge=True,
                norm_feats=True,
                norm_coors=True,
            )
            self.egnn.append(egnn)
            self.norms.append(gnn.norm.GraphNorm(in_channels=example.num_node_features))

        for j in range(fc_layers):
            if j == 0 and fc_layers == 1:
                fc = nn.Linear(8, num_classes)
            elif j + 1 == fc_layers:
                fc = nn.Linear(hidden, num_classes)
            elif j == 0 and fc_layers > 1:
                fc = nn.Linear(8, hidden)
            else:
                fc = nn.Linear(hidden, hidden)
            self.fcs.append(fc)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass over the graph neural network

        Parameters
        ----------
        x : torch.Tensor
            the node features of interest
        edge_index : torch.Tensor
            the indices of the nodes that are contected
        batch : torch.Tensor
            the batch being looked at
        edge_attr : torch.Tensor
            the edge features


        Returns
        -------
        x : torch.Tensor
            the predicted values for each sample

        """
        for i, (egnn, norm) in enumerate(zip(self.egnn, self.norms)):
            x = egnn(
                x=x.to(dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr.to(dtype=torch.float32),
                batch=batch,
            )
            x = norm(x)

            if i + 1 < len(self.egnn):
                x = x.relu().to(torch.float32)

        x = gnn.global_mean_pool(x, batch)

        for j, fc in enumerate(self.fcs):
            print("Expected input size for fc:", fc.weight.shape[1])
            x = fc(x.to(torch.float32))
            x = x.to(torch.float32)

            if j + 1 < self.num_fc:
                x = x.relu()

        return x.to(torch.float32)
