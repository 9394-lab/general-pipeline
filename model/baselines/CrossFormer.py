import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .layers.cross_encoder import Encoder
from .layers.cross_decoder import Decoder
from .layers.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from .layers.cross_embed import DSW_embedding
from math import ceil


class CrossFormer(nn.Module):
    def __init__(self, in_len, out_len, seg_len, edge_index: 'edge_index', edge_weight: 'edge_weights',
                 win_size=4, factor=10, d_model=512, d_ff=1024, n_heads=8, e_layers=3, dropout=0.0,
                 baseline=True, device=torch.device('cuda:0')):
        super(CrossFormer, self).__init__()
        self.name = 'CrossFormer'
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size

        self.baseline = baseline

        self.device = device
        data_dim = edge_index.max().item() + 1
        # The padding operation to handle invisible segment length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, num_nodes=data_dim, block_depth=1, \
                               dropout=dropout, in_seg_num=(self.pad_in_len // seg_len), factor=factor)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, data_dim, dropout, \
                               out_seg_num=(self.pad_out_len // seg_len), factor=factor)

    def forward(self, x_seq):
        if self.baseline:
            base = x_seq[..., 0].mean(dim=1, keepdim=True).permute(0, 2, 1)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if self.in_len_add != 0:
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim=1)

        x_seq = self.enc_value_embedding(x_seq[..., 0].permute(0, 2, 1))
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        output = base + predict_y[:, :self.out_len, :]
        return output.permute(0, 2, 1).unsqueeze(-1)
