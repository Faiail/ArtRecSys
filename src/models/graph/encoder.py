import torch
from torch_geometric.nn import GATConv, to_hetero


class GNNEncoder(torch.nn.Module):
    def __init__(self,
                 hidden_channels=128,
                 num_layers=2,
                 activation=torch.nn.ReLU(),
                 drop_rate=0.4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.drop_rate = drop_rate

        for _ in range(num_layers):
            conv = GATConv((-1, -1), hidden_channels, dropout=drop_rate, add_self_loops=False)
            self.convs.append(conv)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        print(x)
        return x


