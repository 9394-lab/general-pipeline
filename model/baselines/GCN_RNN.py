import torch
from torch import nn
# from SpatialProcessing.GATConv import GATConv
from .DCRNN import ChebDiffusionConvolution

class GLU(nn.Module):
    # Gated Linear Unit
    def __init__(self, input_size, output_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)

class GCN_RNN_block(nn.Module):

    def __init__(self, in_channels, out_channels, edge_index, edge_weight, k, num_layers, batch_norm=False, dropout=0.1):
        nn.Module.__init__(self)

        self.batch_norm = batch_norm

        self.gcn = ChebDiffusionConvolution(in_channels, out_channels,
                                            k=k, edge_index=edge_index, edge_weight=edge_weight)
        self.GLU = GLU(out_channels, out_channels)

        self.rnn = nn.GRU(input_size=out_channels, hidden_size=out_channels, num_layers=num_layers,
                          dropout=dropout, batch_first=True, bidirectional=False)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size * time_len, num_nodes, -1)
        x = self.relu(self.gcn(x))
        x = x.view(batch_size * num_nodes, time_len, -1)

        rnn_out, _ = self.rnn(x)
        rnn_out = self.relu(rnn_out)
        rnn_out = rnn_out.view(batch_size, num_nodes, time_len, -1).contiguous()

        return rnn_out


class GCN_RNN(nn.Module):

    def __init__(self, in_channels, out_channels, num_blocks, input_time_len, output_time_len, num_layers,
                 edge_index: 'edge_index', edge_weight: 'edge_weights', k, dropout):
        nn.Module.__init__(self)
        self.name = 'GCN-RNN'
        self.save_name = self.name
        blockslist = []
        for i in range(num_blocks):
            blockslist.append(GCN_RNN_block(in_channels, out_channels, edge_index, edge_weight, dropout=dropout,
                                            num_layers=num_layers, k=k, batch_norm=True))
            in_channels = out_channels

        self.seq = nn.Sequential(*blockslist)

        self.mlp = nn.Sequential(
            nn.Linear(input_time_len * out_channels, input_time_len),
            nn.ReLU(),
            nn.Linear(input_time_len, output_time_len)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        num_nodes = x.shape[1]

        x = self.seq(x)
        x = x.view(batch_size, num_nodes, -1)
        out = self.mlp(x)
        return out.unsqueeze(-1)
