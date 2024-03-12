import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import math
from math import sqrt


class queryGenerator(nn.Module):
    def __init__(self, heads, dqk, num_nodes, in_channels):
        nn.Module.__init__(self)
        # unified weights
        self.W_u = nn.Parameter(torch.zeros(heads, in_channels, 1, dqk))
        # specific weights
        self.W_s = nn.Parameter(torch.zeros(heads, in_channels, num_nodes, dqk))
        # class weights
        # self.linear = nn.Linear(in_channels * in_time_len, num_class)
        # self.W_c = nn.Parameter(torch.zeros(1, in_channels, num_class, dqk))
        for p in self.W_u, self.W_s:
            nn.init.xavier_normal_(p)

    def forward(self, x):
        num_nodes = x.shape[1]
        # unified x
        W_u = torch.repeat_interleave(self.W_u, num_nodes, dim=2)
        x_u = torch.einsum('bntf, hfnd-> bnhtd', (x, W_u)).contiguous()
        # specific x
        x_s = torch.einsum('bntf, hfnd-> bnhtd', (x, self.W_s)).contiguous()
        return x_u * x_s


def ScaledDotProductAttention(q, k, v, mask=None):
    batch_size = q.shape[0]
    q_time_len = q.shape[-2]
    kv_time_len = v.shape[-2]
    dqk = q.shape[-1]

    att = q.matmul(k) / math.sqrt(dqk)
    att = F.softmax(att, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, 1, q_time_len, kv_time_len).repeat(1, q.shape[1], 1, 1)
        att = att.masked_fill(mask, -1e9)

    out = att.matmul(v)
    out = out.transpose(1, 2).contiguous().view(batch_size, q_time_len, -1)

    return out


class SynAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, dqk=3, heads=4, syns=False):
        nn.Module.__init__(self)
        self.num_nodes = num_nodes
        self.heads = heads
        self.Sync = syns

        if syns:
            self.queryGene = queryGenerator(heads, dqk, num_nodes, in_channels)
            self.W_k = nn.Parameter(torch.zeros(heads, in_channels, 1, dqk))
            nn.init.xavier_normal_(self.W_k)
        else:
            self.W_q = nn.Parameter(torch.zeros(heads, in_channels, dqk))
            self.W_k = nn.Parameter(torch.zeros(heads, in_channels, dqk))
            nn.init.xavier_normal_(self.W_q)
            nn.init.xavier_normal_(self.W_k)
        self.W_v = nn.Parameter(torch.zeros(heads, in_channels, out_channels))
        nn.init.xavier_normal_(self.W_v)

    def forward(self, x, mask=None):
        r"""
        :param x: (B, N, T, F)
        :param mask: (B, T, T)
        :return: out: (B, N, T, F)
        """
        batch_size, num_nodes, time_len, input_fea = x.shape
        if self.Sync:
            q = self.queryGene(x)
            q = q.view(batch_size * num_nodes, self.heads, time_len, -1)
            W_k = torch.repeat_interleave(self.W_k, num_nodes, dim=2)
            k = torch.einsum('bntf, hfnd-> bnhtd', (x, W_k)).contiguous()
            k = k.view(batch_size * num_nodes, self.heads, -1, time_len)
            x = x.contiguous().view(batch_size * num_nodes, 1, time_len, -1)
            v = torch.einsum('batf, hfe-> bhte', (x, self.W_v)).contiguous()
        else:
            q = torch.einsum('bntf, hfd-> bnhtd', (x, self.W_q)).contiguous()
            q = q.view(batch_size * num_nodes, self.heads, time_len, -1)
            k = torch.einsum('bntf, hfd-> bnhtd', (x, self.W_k)).contiguous()
            k = k.view(batch_size * num_nodes, self.heads, -1, time_len)
            x = x.contiguous().view(batch_size * num_nodes, 1, time_len, -1)
            v = torch.einsum('batf, hfe-> bhte', (x, self.W_v)).contiguous()
        out = ScaledDotProductAttention(q, k, v, mask)
        out = out.contiguous().view(batch_size*num_nodes, time_len, input_fea)
        return out


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1, num_nodes=1362):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        # self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.time_attention = SynAttention(d_model, int(d_model / n_heads), num_nodes, dqk=3, heads=n_heads, syns=True)
        self.bridge_sender = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.bridge_receiver = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.deck = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

        self.proj = nn.Linear(d_model*2, d_model)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # B N T F
        batch, num_nodes, time_len, features = x.shape
        # 改变维度得到时空输入
        st_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        # geographic-heterogeneity temporal attention
        time_enc = self.time_attention(x)
        time_enc = st_in + self.dropout(time_enc)
        time_enc = self.norm1(time_enc)
        time_enc = time_enc + self.dropout(self.MLP1(time_enc))
        time_enc = self.norm2(time_enc).view(batch, num_nodes, time_len, -1)

        # Bridge spatial attention: bridge deck（桥面）
        spa_in = rearrange(st_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        bridge_deck = repeat(self.deck, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        deck_buffer = self.bridge_sender(bridge_deck, spa_in, spa_in)
        spa_enc = self.bridge_receiver(spa_in, deck_buffer, deck_buffer)
        spa_enc = spa_in + self.dropout(spa_enc)
        spa_enc = self.norm3(spa_enc)
        spa_enc = spa_enc + self.dropout(self.MLP2(spa_enc))
        spa_enc = self.norm4(spa_enc).view(batch, num_nodes, time_len, -1)

        # 时空分别提取并concat 减少过拟合情况
        out = self.proj(torch.cat([time_enc, spa_enc], dim=-1))
        out = self.proj_drop(out)
        return out

    # def forward(self, x):
    #     # Cross Time Stage: Directly apply MSA to each dimension
    #     # B N T F
    #     batch = x.shape[0]
    #     time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
    #     time_enc = self.time_attention(x)
    #     dim_in = time_in + self.dropout(time_enc)
    #     dim_in = self.norm1(dim_in)
    #     dim_in = dim_in + self.dropout(self.MLP1(dim_in))
    #     dim_in = self.norm2(dim_in)
    #
    #     # Cross Dimension Stage: use a small set of learnable vectors to aggregate
    #     # and distribute messages to build the D-to-D connection
    #     dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
    #     batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
    #     dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
    #     dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
    #     dim_enc = dim_send + self.dropout(dim_receive)
    #     dim_enc = self.norm3(dim_enc)
    #     dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
    #     dim_enc = self.norm4(dim_enc)
    #
    #     final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)
    #
    #     return final_out


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout = 0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)


# class TwoStageAttentionLayer(nn.Module):
#     '''
#     The Two Stage Attention (TSA) Layer
#     input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
#     '''
#     def __init__(self, seg_num, factor, d_model, n_heads, d_ff = None, dropout=0.1):
#         super(TwoStageAttentionLayer, self).__init__()
#         d_ff = d_ff or 4*d_model
#         self.time_attention = AttentionLayer(d_model, n_heads, dropout = dropout)
#         self.dim_sender = AttentionLayer(d_model, n_heads, dropout = dropout)
#         self.dim_receiver = AttentionLayer(d_model, n_heads, dropout = dropout)
#         self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
#
#         self.dropout = nn.Dropout(dropout)
#
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.norm4 = nn.LayerNorm(d_model)
#
#         self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
#                                 nn.GELU(),
#                                 nn.Linear(d_ff, d_model))
#         self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
#                                 nn.GELU(),
#                                 nn.Linear(d_ff, d_model))
#
#     def forward(self, x):
#         #Cross Time Stage: Directly apply MSA to each dimension
#         batch = x.shape[0]
#         time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
#         time_enc = self.time_attention(
#             time_in, time_in, time_in
#         )
#         dim_in = time_in + self.dropout(time_enc)
#         dim_in = self.norm1(dim_in)
#         dim_in = dim_in + self.dropout(self.MLP1(dim_in))
#         dim_in = self.norm2(dim_in)
#
#         #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
#         dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
#         batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)
#         dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
#         dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
#         dim_enc = dim_send + self.dropout(dim_receive)
#         dim_enc = self.norm3(dim_enc)
#         dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
#         dim_enc = self.norm4(dim_enc)
#
#         final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)
#
#         return final_out
