"""GNN model."""

import torch
import torch_geometric
from torch import nn
from torch_geometric import nn as gnn


def edge_model(input_size: int, output_size: int, num_layers: int):
    """NN model used to process the edge feautures for the main graph neural network.

    Parameters
    ----------
    input_size : int
        the input size for the NN
    output_size : int
        the output size for the NN
    num_layers : int
        number of layers in the fc for the edge features

    Returns
    -------
    fully connected nn model : object, the list of the layers as a torch sequential obj

    """
    layers = []

    for i in range(num_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(output_size, output_size))
        layers.append(nn.ReLU())

    result = nn.Sequential(*layers)
    # print(f"edge_model: {result}, {type(result)}")

    return result


class MP_model(nn.Module):
    """Graph Neural Network using the MPNN layers."""

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        num_classes: int,
        mp_layers: int = 3,
        edge_layers: int = 3,
        fcc_in_gnn: int = 5,
        aggr: str = "mean",
        hidden: int = 32,
        dropout: float = 0,
        dropnode: float = 0,
        graphnorm: bool = False,
    ) -> None:
        """Initalizing the model.

        Parameters
        ----------
        num_node_features: int
            number of node feaures
        num_edge_features: int
            nmber of edge features
        num_classes: int
            number of the targets (therefore size of the output layer)
        mp_layers: int
            dafault=3 , number of message passing layers
        edge_layers: int
            default=3, number of fcc layers in the edge model
        fcc_in_gnn: int
            default=5, number of the fully connected layers after the message passing
        aggr: str
            default=mean, the aggration function
        hidden: int
            default=32, the number of nodes in the hidden layers
        dropout: float
            default=0
        dropnode: float
            default=0
        graphnorm: bool
            deafult=True

        Return
        ------
        None

        """
        super(MP_model, self).__init__()

        self.mps = mp_layers
        self.num_fc = fcc_in_gnn
        self.dropout = nn.Dropout(p=dropout)
        self.dropnode = dropnode
        self.graphnorm = graphnorm

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if graphnorm else None
        self.fcs = nn.ModuleList()

        for i in range(mp_layers):
            if i == 0:
                conv = gnn.conv.NNConv(
                    in_channels=num_node_features,
                    out_channels=hidden,
                    aggr=aggr,
                    nn=edge_model(
                        num_edge_features, (num_node_features * hidden), edge_layers
                    ),
                )
            else:
                conv = gnn.conv.NNConv(
                    in_channels=hidden,
                    out_channels=hidden,
                    aggr=aggr,
                    nn=edge_model(num_edge_features, (hidden**2), edge_layers),
                )
            self.convs.append(conv)

            if self.graphnorm:
                self.norms.append(gnn.norm.GraphNorm(in_channels=hidden))

        for j in range(fcc_in_gnn):
            if j + 1 == fcc_in_gnn:
                fc = nn.Linear(hidden, num_classes)
            else:
                fc = nn.Linear(hidden, hidden)
            self.fcs.append(fc)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass over the graph neural network.

        Parameters
        ----------
        x : torch tensor
            the node features of interest
        edge_index : torch tensor
            the indices of the nodes that are contected
        edge_attr : torch tensor
            the edge features
        batch : torch tensor
            the batch being looked at

        Returns
        -------
        x : torch tensor
            the predicted values for each sample

        """
        if self.graphnorm:
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                if self.dropnode > 0:
                    edge_index, edge_mask, node_mask = (
                        torch_geometric.utils.dropout_node(edge_index, p=self.dropnode)
                    )
                    x = x[node_mask]
                    edge_attr = edge_attr[edge_mask]

                x = conv(x.to(torch.float32), edge_index, edge_attr.to(torch.float32))
                x = norm(x)
                x = x.to(torch.float32)
            if i + 1 < self.mps:
                x.relu().to(torch.float32)

        else:
            for i, conv in enumerate(self.convs):
                if self.dropnode > 0:
                    edge_index, edge_mask, node_mask = (
                        torch_geometric.utils.dropout_node(edge_index, p=self.dropnode)
                    )
                    x = x[node_mask]
                    edge_attr = edge_attr[edge_mask]

                x = conv(x.to(torch.float32), edge_index, edge_attr.to(torch.float32))
                x = x.to(torch.float32)
                if i + 1 < self.mps:
                    x.relu().to(torch.float32)

        x = gnn.global_mean_pool(x, batch)

        for j, fc in enumerate(self.fcs):
            x = fc(x.to(torch.float32))
            x = x.to(torch.float32)
            if j + 1 < self.num_fc:
                x = self.dropout(x)
                x = x.relu()

        return x.to(torch.float32)
