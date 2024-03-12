"""
PROJECT_NAME:  general-traffic-tasks-framework ; 
FILE_NAME: LTB ;
CreatTime: 2024/3/7;
AUTHOR: 93943 ;
E_MAIL: tguan@zju.edu.cn
"""
from .LTBModules import *
from .BridgeModules import *
import torch
import torch.nn as nn


class LTB(nn.Module):
    def __init__(self, edge_index: 'edge_index', in_len=12, out_len=12, seg_len=4,
                 factor=10, d_model=512, d_ff=1024, n_heads=8, num_blocks=4, stride=1,
                 dropout=0.0, syns=True, add_base=True):
        super(LTB, self).__init__()
        self.name = 'LTB_num_blocks_{}_dmodel_{}_dff_{}_syns_{}_seglen_{}_stride_{}'\
            .format(num_blocks, d_model, d_ff, syns, seg_len, stride)
        num_nodes = edge_index.max().item() + 1
        # num_nodes = 307
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.stride = stride
        self.add_base = add_base

        # padding
        self.padding = False
        self.pad_len = 0
        if (in_len - seg_len) % stride != 0:
            self.pad_len = stride - ((in_len - seg_len) % stride)
            self.padding = True
        patch_num = (in_len + self.pad_len - seg_len) // stride + 1

        # Embedding
        self.enc_value_embedding = temporal_embedding(patch_num, seg_len, stride, d_model, num_nodes)  # patch_num, num_nodes, d_model
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, num_nodes, patch_num, d_model))  # 1, num_nodes, patch_num, d_model
        self.pre_norm = nn.LayerNorm(d_model)  # d_model,

        # Encoder
        self.encoder = Encoder(num_blocks, d_model, n_heads, d_ff,
                               dropout, num_nodes, patch_num, factor=factor, syns=syns)

        self.output_layers = nn.ModuleList()
        for i in range(num_blocks + 1):
            self.output_layers.append(nn.Linear(patch_num, out_len))

        self.end = nn.Linear(d_model, 1)

    def forward(self, x_seq):
        # B N T F -> B T N 和 B N T tod+dow
        x_seq, time_seq = x_seq[..., 0], x_seq[..., 1:]
        x_seq = x_seq.squeeze(-1).permute(0, 2, 1)

        # add_base
        if self.add_base:
            base = x_seq.mean(dim=1, keepdim=True).permute(0, 2, 1).unsqueeze(-1)  # B N 1 1
        else:
            base = 0

        # padding B Tpadding N 和 B N Tpadding tod+dow
        if self.padding:
            x_seq = torch.cat((x_seq[:, [-1], :].expand(-1, self.pad_len, -1), x_seq), dim=1)
            time_seq = torch.cat((time_seq[:, :, [-1], :].expand(-1, -1, self.pad_len, -1), time_seq), dim=2)

        # embedding
        x_seq = self.enc_value_embedding(x_seq, time_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        # encoder
        enc_out = self.encoder(x_seq)

        # generate output
        output = 0
        for i in range(len(enc_out)):
            output += self.output_layers[i](enc_out[i].permute(0, 1, 3, 2))
        output = output.permute(0, 1, 3, 2)
        output = self.end(output)  # batch_size, num_nodes, out_len, 1

        return output + base