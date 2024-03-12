import torch
from torch import nn


class mlp(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, in_time_len=12, out_time_len=12, out_channels=3):
        nn.Module.__init__(self)

        self.name='mlp1'

        self.m = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.out_ln = nn.Linear(hidden_channels*in_time_len, out_time_len*out_channels)

        self.out_time_len = out_time_len
        # self.ln = nn.Linear(1, 3)

    def forward(self, x):
        # x: b n  t f

        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        # time_len = x.shape[2]
        # fea_num = x.shape[3]

        h = self.m(x)
        h = h.view(batch_size, num_nodes, -1)
        out = self.out_ln(h)
        out = out.view(batch_size, num_nodes, self.out_time_len, -1)

        return out
