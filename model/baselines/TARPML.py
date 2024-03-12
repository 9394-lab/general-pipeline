#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @date: 2021-01-22 21:09
# @name: TARPML
# @author：Peng-pai

import torch.nn as nn
import torch


class TARPML(nn.Module):
    def __init__(self, in_channels, out_channels, input_time_len, output_time_len, weather_dimension, num_layers=4, dropout=0.1):
        nn.Module.__init__(self)
        self.name = 'TARPML'
        self.in_channels = in_channels
        self.start_conv = nn.Linear(in_channels, out_channels)
        self.rnn = nn.LSTM(input_size=in_channels, hidden_size=out_channels, num_layers=num_layers,
                           dropout=dropout, batch_first=True, bidirectional=False)
        self.embedding = nn.Embedding(weather_dimension, 8)
        self.FC = nn.Sequential(nn.Linear(out_channels, out_channels * 2),
                                nn.ReLU(),
                                nn.Linear(out_channels * 2, out_channels * 2),
                                nn.ReLU(),
                                nn.Linear(out_channels * 2, out_channels * 2),
                                nn.ReLU(),
                                )
        self.endconv = nn.Linear(out_channels * 2 * input_time_len, output_time_len)
        self.endConv2 = nn.Linear(1, 3)

    def forward(self, x):
        # x: B N T F_out
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]

        x = x[..., :self.in_channels].view(batch_size * num_nodes, time_len, -1)

        rnn_out, _ = self.rnn(x)
        out = self.FC(rnn_out)
        out = self.endconv(out.contiguous().view(batch_size, num_nodes, -1))
        out = self.endConv2(out.unsqueeze(dim=-1))
        return out
