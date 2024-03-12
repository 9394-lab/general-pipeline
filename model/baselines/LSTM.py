import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_channels, out_channels, input_time_len, output_time_len, num_layers=2, dropout=0):
        nn.Module.__init__(self)
        self.name = 'LSTM1'
        self.save_name = self.name
        self.in_channels = in_channels
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=out_channels, num_layers=num_layers,
                           dropout=dropout, batch_first=True, bidirectional=False)
        self.Ln = nn.Sequential(nn.Linear(out_channels, out_channels),
                                nn.ReLU(),)
        self.endconv = nn.Linear(input_time_len*out_channels, output_time_len)

    def forward(self, x):
        # input: B N T F
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]

        rnn_out, _ = self.rnn(x.view(batch_size*num_nodes, time_len, -1))
        # rnn_out = rnn_out[:, -1, :]
        rnn_out = rnn_out.contiguous().view(batch_size, num_nodes, time_len, -1)  # B N T F
        out = self.Ln(rnn_out).contiguous().view(batch_size, num_nodes, -1)
        out = self.endconv(out).unsqueeze(-1)
        return out
