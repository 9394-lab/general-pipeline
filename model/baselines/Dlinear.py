import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.PatchTST_backbone import series_decomp


class Dlinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, seq_len, pred_len, edge_index: 'edge_index', type, moving_avg=25, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Dlinear, self).__init__()
        self.name = type + 'linear'
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.decompsition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = edge_index.max().item() + 1

        if type == 'D':
            if self.individual:
                self.Linear_Seasonal = nn.ModuleList()
                self.Linear_Trend = nn.ModuleList()

                for i in range(self.channels):
                    self.Linear_Seasonal.append(
                        nn.Linear(self.seq_len, self.pred_len))
                    self.Linear_Trend.append(
                        nn.Linear(self.seq_len, self.pred_len))

                    self.Linear_Seasonal[i].weight = nn.Parameter(
                        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                    self.Linear_Trend[i].weight = nn.Parameter(
                        (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            else:
                self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
                self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

                self.Linear_Seasonal.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            if self.individual:
                self.Linear = nn.ModuleList()
                for i in range(self.channels):
                    self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
            else:
                self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def Dlinear(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def Nlinear(self, x ):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x  # [Batch, Output length, Channel]

    def forward(self, x_enc):
        x_enc = x_enc[..., [0]].squeeze(-1).permute(0, 2, 1)
        if self.name == 'Dlinear':
            dec_out = self.Dlinear(x_enc)
        else:
            if self.name == 'Nlinear':
                seq_last = x_enc[:, -1:, :].detach()
                x_enc = x_enc - seq_last
                dec_out = self.Nlinear(x_enc) + seq_last
            else:
                dec_out = self.Nlinear(x_enc)
        return dec_out[:, -self.pred_len:, :].permute(0, 2, 1).unsqueeze(-1)  # [B, L, D]