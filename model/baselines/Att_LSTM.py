import torch
from torch import nn
import numpy as np
"""
Short-Term Traffic Congestion Forecasting Using Attention-Based Long Short-Term Memory Recurrent Neural Network
"""


class Att_LSTM(nn.Module):

    def __init__(self, in_channels, out_channels, input_time_len, output_time_len, num_layers=2, dropout=0):
        nn.Module.__init__(self)
        self.name = 'Att_LSTM'
        self.in_channels = in_channels
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=out_channels, num_layers=num_layers,
                           dropout=dropout, batch_first=True)
        self.Ln = nn.Sequential(nn.Linear(out_channels, out_channels),
                                nn.ReLU(),)
        self.endconv = nn.Linear(input_time_len*out_channels, output_time_len)
        self.end1 = nn.Linear(1, 3)

    def forward(self, x):
        # input: B N T F
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]

        x = x[..., :self.in_channels]
        rnn_out, _ = self.rnn(x.view(batch_size*num_nodes, time_len, -1))
        rnn_out = rnn_out.contiguous().view(batch_size, num_nodes, time_len, -1)  # B N T F
        rnn_out = self.Ln(rnn_out)
        u_t = torch.tanh(rnn_out)
        out = rnn_out * u_t
        out = self.endconv(out.contiguous().view(batch_size, num_nodes, -1))
        out = self.end1(out.unsqueeze(dim=-1))
        return out
