import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class temporal_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(temporal_embedding, self).__init__()
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

        self.minute_size = 288
        self.daytime_embedding = nn.Embedding(self.minute_size, d_model)
        weekday_size = 7
        self.weekday_embedding = nn.Embedding(weekday_size, d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_tem):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)
        # 时间维度应该不能为0
        if x_tem.shape[-1] != 0:
            x_embed += self.daytime_embedding(x_tem[:, :, :, 0].long())
            x_embed += self.weekday_embedding(x_tem[:, :, :, 1].long())
        x = self.dropout(x_embed)
        return x


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)
        return x_embed
