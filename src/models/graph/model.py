import torch
from .encoder import GNNEncoder
from torch_geometric.nn import to_hetero


class ContextAwareArtRecSys(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, gnn_layers, activation):
        super(ContextAwareArtRecSys, self).__init__()
        self.encoder = GNNEncoder(num_layers=gnn_layers, activation=activation, hidden_channels=hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata=metadata)
        self.lin_user = torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels//2)
        self.lin_item = torch.nn.Linear(in_features=hidden_channels, out_features=hidden_channels//2)
        self.out_lin = torch.nn.Linear(in_features=self.lin_user.out_features*2, out_features=1)
        self.activation = activation

    def forward(self, graph, entries):
        z_dict = self.encoder(graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict)
        users = z_dict['user'][entries[0]]
        items = z_dict['artwork'][entries[1]]
        user_feats = self.lin_user(users)
        item_feats = self.lin_item(items)
        return self.out_lin(torch.cat([user_feats, item_feats], axis=1))



