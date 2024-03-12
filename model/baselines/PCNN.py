import torch
from torch import nn
import torch.nn.functional as F


class PCNN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, input_time_len, output_time_len, kernel_size=1,
                 num_layers=4, dropout=0.2):
        nn.Module.__init__(self)
        self.name = 'PCNN'
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.begin_conv = nn.Linear(in_channels, hidden_channels)
        self.cnn = nn.ModuleList([nn.Conv2d(hidden_channels, hidden_channels, (1, kernel_size)) for _ in range(num_layers)])
        self.ln = nn.Conv2d(hidden_channels, out_channels, (1, kernel_size))
        self.endconv = nn.Linear(input_time_len * out_channels, output_time_len)
        self.end1 = nn.Linear(1, 3)

    def forward(self, x):
        # B N T F
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        x = x[..., :self.in_channels]  # B N T 1

        cnn_out = self.begin_conv(x).permute(0, 3, 2, 1)
        cnn_out = F.dropout(cnn_out, self.dropout)
        for i in range(self.num_layers):
            cnn_out = self.cnn[i](cnn_out)

        out = self.ln(cnn_out)
        out = self.endconv(out.contiguous().view(batch_size, num_nodes, -1))
        out = self.end1(out.unsqueeze(dim=-1))
        return out
