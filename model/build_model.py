import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import configparser
import os.path as osp
from model.mymodels import *
from model.baselines import *
# from model.baselines.DCRNN import ChebDiffusionConvolution, DiffusionConvolution
from torchinfo import summary


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, out_channels, dropout, max_len=12):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, out_channels)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, out_channels, 2) *
                             -(math.log(10000.0) / out_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(2)]
        return self.dropout(x)


class Embedding_layer(nn.Module):
    def __init__(self, in_channels, dropout, in_time_len):
        nn.Module.__init__(self)
        self.linear = nn.Linear(1, in_channels)

        minute_size = 288
        self.daytime_embedding = nn.Embedding(minute_size, in_channels)
        weekday_size = 7
        self.weekday_embedding = nn.Embedding(weekday_size, in_channels)
        self.dropout = nn.Dropout(0.2)

        self.pe = PositionalEncoding(in_channels, dropout=dropout, max_len=in_time_len)

    def forward(self, x):
        x_embed = self.linear(x[..., [0]])
        if x.shape[-1] > 1:
            time_emb = self.daytime_embedding(x[:, :, :, 1].long())
            day_emb = self.weekday_embedding(x[:, :, :, 2].long())
            x_embed = x_embed + time_emb + day_emb
        x_embed = self.dropout(x_embed)
        x_embed = self.pe(x_embed)
        return x_embed


def build_model(datasets, model_args, tpe_args=None, backbone=None):
    cfg = configparser.ConfigParser()
    cfg.read('./config/model/{}.cfg'.format(model_args))

    kwargs = {}
    print('==' * 25)
    print('begin import model cfg file...')
    for section in cfg.sections():
        if section != 'build':
            for val in cfg.items(section):
                expr = 'kwargs[\'{}\']={}'.format(val[0], val[1])
                exec(expr)
                print('{:20}\t{}'.format(val[0], val[1]))

    # 方便Transfer改变backbone
    if backbone is not None:
        kwargs['backbone'] = backbone

    # 用tpe_args替换当前的kwargs对应的args
    if tpe_args:
        print('begin auto-change hyper-parameters...')
        for key in tpe_args:
            expr = 'kwargs[\'{}\']={}'.format(key, tpe_args[key])
            exec(expr)
            print('{:20}\t{}'.format(key, tpe_args[key]))

    model_fun = eval(cfg.get('build', 'func'))
    print('end')
    print('==' * 25)

    annotations_dict = model_fun.__init__.__annotations__
    for key in annotations_dict:
        if annotations_dict[key] == 'edge_index':
            kwargs[key] = datasets[0].edge_index
        if annotations_dict[key] == 'edge_weights':
            kwargs[key] = datasets[0].edge_weights
        if annotations_dict[key] == 'num_nodes':
            kwargs[key] = datasets[0].num_nodes

    model = model_fun(**kwargs)
    # summary(
    #     model,
    #     [
    #         cfg["batch_size"],
    #         cfg["num_nodes"],
    #         cfg["in_len"],
    #         cfg[""],
    #         ,
    #     ],
    #     verbose=0,  # avoid print twice
    # )

    return model
