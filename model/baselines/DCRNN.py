import torch
from torch import nn
import numpy as np
import torch.utils.data

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

class DiffusionConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, k, edge_index, edge_weight, bidirectional=True, remove_ln=False):
        MessagePassing.__init__(self, aggr='add')

        self.num_nodes = edge_index.max() + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.bidirectional = bidirectional
        self.remove_ln = remove_ln

        self.edge_index = edge_index
        deg = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=self.num_nodes)
        self.diffusion_edge_weight = (edge_weight / deg[edge_index[0]]).view(-1, 1)
        self.edge_index = nn.Parameter(self.edge_index, requires_grad=False)
        self.diffusion_edge_weight = nn.Parameter(self.diffusion_edge_weight, requires_grad=False)

        if remove_ln:
            remove_param_num = 1
        else:
            remove_param_num = 0

        if self.bidirectional:
            deg = scatter_add(edge_weight, edge_index[1], dim=0, dim_size=self.num_nodes)
            self.re_diffusion_edge_weight = (edge_weight / deg[edge_index[1]]).view(-1, 1)
            self.re_diffusion_edge_weight = nn.Parameter(self.re_diffusion_edge_weight, requires_grad=False)
            self.param = nn.Parameter(torch.ones((2 * k - 1 - remove_param_num) * in_channels, out_channels))
        else:
            self.param = nn.Parameter(torch.ones((k - remove_param_num) * in_channels, out_channels))

        nn.init.xavier_uniform_(self.param)

        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.constant_(self.bias, 0)

    def forward(self, x):

        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        features_num = x.shape[2]

        conv_out = x.transpose(0, 1).contiguous().view(num_nodes, batch_size * features_num)
        reconv_out = conv_out

        conv_out_list = [conv_out]

        for i in range(1, self.k):
            conv_out = self.propagate(torch.stack([self.edge_index[0], self.edge_index[1]], dim=0), x=conv_out,
                                      edge_weight=self.diffusion_edge_weight)
            conv_out_list.append(conv_out)

            if self.bidirectional:
                reconv_out = self.propagate(torch.stack([self.edge_index[1], self.edge_index[0]], dim=0), x=reconv_out,
                                            edge_weight=self.re_diffusion_edge_weight)
                conv_out_list.append(reconv_out)

        if self.remove_ln:
            conv_out_list = conv_out_list[1:]

        out = torch.stack(conv_out_list, dim=2)

        # if self.bidirectional:
        #     out = out.view(num_nodes * batch_size, (2 * self.k - 1) * features_num)
        # else:
        #     out = out.view(num_nodes * batch_size, self.k * features_num)

        out = out.view(num_nodes * batch_size, -1)

        out = out.matmul(self.param).view(num_nodes, batch_size, -1).transpose(0, 1).contiguous()

        return out + self.bias

    def message(self, x_j, edge_weight):
        return x_j * edge_weight


class ChebDiffusionConvolution(DiffusionConvolution):

    def forward(self, x):

        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        features_num = x.shape[2]

        x = x.transpose(0, 1).contiguous().view(num_nodes, batch_size * features_num)

        conv_x0 = x
        reconv_x0 = conv_x0

        conv_x1 = self.propagate(torch.stack([self.edge_index[0], self.edge_index[1]], dim=0), x=conv_x0,
                                 edge_weight=self.diffusion_edge_weight)

        if self.bidirectional:
            reconv_x1 = self.propagate(torch.stack([self.edge_index[1], self.edge_index[0]], dim=0), x=reconv_x0,
                                       edge_weight=self.re_diffusion_edge_weight)
            conv_out_list = [conv_x0, conv_x1, reconv_x1]
        else:
            conv_out_list = [conv_x0, conv_x1]

        for i in range(2, self.k):
            conv_out = self.propagate(torch.stack([self.edge_index[0], self.edge_index[1]], dim=0), x=conv_x1,
                                      edge_weight=self.diffusion_edge_weight)
            conv_out = 2 * conv_out - conv_x0
            conv_out_list.append(conv_out)
            conv_x0 = conv_x1
            conv_x1 = conv_out

            if self.bidirectional:
                reconv_out = self.propagate(torch.stack([self.edge_index[1], self.edge_index[0]], dim=0), x=reconv_x1,
                                            edge_weight=self.re_diffusion_edge_weight)
                reconv_out = 2 * reconv_out - reconv_x0
                conv_out_list.append(reconv_out)
                reconv_x0 = reconv_x1
                reconv_x1 = reconv_out

        if self.remove_ln:
            conv_out_list = conv_out_list[1:]

        out = torch.stack(conv_out_list, dim=2)

        # if self.bidirectional:
        #     out = out.view(num_nodes * batch_size, (2 * self.k - 1) * features_num)
        # else:
        #     out = out.view(num_nodes * batch_size, self.k * features_num)

        out = out.view(num_nodes * batch_size, -1)

        out = out.matmul(self.param).view(num_nodes, batch_size, -1).transpose(0, 1).contiguous()

        return out + self.bias


class DCGRUcell(nn.Module):
    def __init__(self, in_channels, out_channels, conv_layer=ChebDiffusionConvolution, **kwargs):
        nn.Module.__init__(self)

        self.out_channels = out_channels

        self.ru_dc = conv_layer(in_channels + out_channels, 2 * out_channels, **kwargs)
        self.c_dc = conv_layer(in_channels + out_channels, out_channels, **kwargs)

        self.out_channels = out_channels

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = torch.zeros(x.shape[0], x.shape[1], self.out_channels, device=x.device)

        x_and_state = torch.cat([x, hidden_state], dim=2)

        ru = torch.sigmoid(self.ru_dc(x_and_state))
        r = ru[:, :, :self.out_channels]
        u = ru[:, :, self.out_channels:]

        rh = r.mul(hidden_state)
        xrh = torch.cat([x, rh], dim=-1)

        c = torch.tanh(self.c_dc(xrh))
        c = torch.tanh(c)

        out = u.mul(hidden_state) + (1 - u).mul(c)

        return out


class Endecoder(nn.Module):
    def __init__(self, in_channels, out_channels_list, **kwargs):
        nn.Module.__init__(self)

        self.dcgrulist = nn.ModuleList([DCGRUcell(in_channels, out_channels_list[0], **kwargs)])
        for i in range(1, len(out_channels_list)):
            self.dcgrulist = self.dcgrulist.append(DCGRUcell(out_channels_list[i - 1], out_channels_list[i], **kwargs))

    def forward(self, x, hidden_state=None):

        if hidden_state is None:
            hidden_state = [None] * len(self.dcgrulist)
        hidden_state_list = []

        h = x
        for i, m in enumerate(self.dcgrulist):
            h = m(h, hidden_state[i])
            hidden_state_list.append(h)

        return hidden_state_list


class DCRNN(nn.Module):
    def __init__(self, encoder_in_channels, hidden_channels_list, out_channels, output_time_len,
                 edge_index: 'edge_index', edge_weight: 'edge_weights',
                 scheduled_sampling_param=0, **kwargs):

        nn.Module.__init__(self)

        self.name = 'DCRNN'
        self.save_name = 'DCRNN'

        self.output_time_len = output_time_len

        self.encoder = Endecoder(encoder_in_channels, hidden_channels_list, edge_index=edge_index,
                                 edge_weight=edge_weight, **kwargs)
        self.decoder = Endecoder(out_channels, hidden_channels_list, edge_index=edge_index, edge_weight=edge_weight,
                                 **kwargs)

        self.FC = nn.Linear(hidden_channels_list[-1], out_channels)
        self.scheduled_sampling_param = scheduled_sampling_param
        self.__input_samples_num = 0

        self.out_channels = out_channels

    def reset_input_samples_num(self):
        self.__input_samples_num = 0

    def add_input_samples_num(self, value):
        self.__input_samples_num += value

    def forward(self, x, y=None):
        if y is None:
            y = x[..., 0]

        x = x[..., [0]]
        hidden_state = None
        for i in range(x.shape[2]):
            hidden_state = self.encoder(x[:, :, i, :], hidden_state)

        out_list = []
        out = torch.zeros(y.shape[0], y.shape[1], self.out_channels, device=x.device)

        for i in range(self.output_time_len):
            hidden_state = self.decoder(out, hidden_state)
            out = self.FC(hidden_state[-1])
            out_list.append(out)

            if self.training and (self.scheduled_sampling_param > 0):
                true_proba = self.scheduled_sampling_param / (self.scheduled_sampling_param + np.exp(
                    self.__input_samples_num / self.scheduled_sampling_param))
                r_v = np.random.uniform(0, 1)
                if r_v < true_proba:
                    out = y[:, :, i].view(x.shape[0], x.shape[1], self.out_channels)

        re = torch.stack(out_list, dim=2)
        re = re.view(y.shape[0], y.shape[1], self.output_time_len, self.out_channels)

        return re
