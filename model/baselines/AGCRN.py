# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AVWGCN(nn.Module):
#     def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
#         super(AVWGCN, self).__init__()
#         self.cheb_k = cheb_k
#         self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
#         self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
#
#     def forward(self, x, node_embeddings):
#         # x shaped[B, N, C], node_embeddings shaped [N, D, 3] -> supports shaped [N, N]
#         # output shape [B, N, C]
#         i, j = 0, -1
#         if node_embeddings.shape[-1] == 1:
#             i, j = 0, 0
#         node_num = node_embeddings.shape[0]
#         supports = F.softmax(F.relu(torch.mm(node_embeddings[..., i], node_embeddings[..., j].transpose(0, 1))), dim=1)
#         support_set = [torch.eye(node_num).to(supports.device), supports]
#         # default cheb_k = 3
#         for k in range(2, self.cheb_k):
#             support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
#         supports = torch.stack(support_set, dim=0)
#         weights = torch.einsum('nd,dkio->nkio', node_embeddings[..., -1], self.weights_pool)  # N, cheb_k, dim_in, dim_out
#         bias = torch.matmul(node_embeddings[..., -1], self.bias_pool)  # N, dim_out
#         x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
#         x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
#         x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
#         return x_gconv
#
#
# class AGCRNCell(nn.Module):
#     def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
#         super(AGCRNCell, self).__init__()
#         self.node_num = node_num
#         self.hidden_dim = dim_out
#         self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
#         self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)
#
#     def forward(self, x, state, node_embeddings):
#         # x: B, num_nodes, input_dim
#         # state: B, num_nodes, hidden_dim
#         state = state.to(x.device)
#         input_and_state = torch.cat((x, state), dim=-1)
#         z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
#         z, r = torch.split(z_r, self.hidden_dim, dim=-1)
#         candidate = torch.cat((x, z * state), dim=-1)
#         hc = torch.tanh(self.update(candidate, node_embeddings))
#         h = r * state + (1 - r) * hc
#         return h
#
#     def init_hidden_state(self, batch_size):
#         return torch.zeros(batch_size, self.node_num, self.hidden_dim)
#
#
# class AVWDCRNN(nn.Module):
#     def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
#         super(AVWDCRNN, self).__init__()
#         assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
#         self.node_num = node_num
#         self.input_dim = dim_in
#         self.num_layers = num_layers
#         self.dcrnn_cells = nn.ModuleList()
#         self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
#         for _ in range(1, num_layers):
#             self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))
#
#     def forward(self, x, init_state, node_embeddings):
#         # shape of x: (B, T, N, D)
#         # shape of init_state: (num_layers, B, N, hidden_dim)
#         assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
#         seq_length = x.shape[1]
#         current_inputs = x
#         output_hidden = []
#         for i in range(self.num_layers):
#             state = init_state[i]
#             inner_states = []
#             for t in range(seq_length):
#                 state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
#                 inner_states.append(state)
#             output_hidden.append(state)
#             current_inputs = torch.stack(inner_states, dim=1)
#         # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
#         # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
#         # last_state: (B, N, hidden_dim)
#         return current_inputs, output_hidden
#
#     def init_hidden(self, batch_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
#         return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)
#
#
# class AGCRN(nn.Module):
#     def __init__(self, edge_index: 'edge_index', input_dim, rnn_units, output_dim, horizon,
#                  num_layers, embed_dim, cheb_k, dropout):
#         super(AGCRN, self).__init__()
#
#         self.num_nodes = edge_index.max().item() + 1
#         self.input_dim = input_dim
#         self.hidden_dim = rnn_units
#         self.output_dim = output_dim
#         self.horizon = horizon
#         self.num_layers = num_layers
#
#         self.name = 'AGCRN'
#
#         self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, embed_dim, 2), requires_grad=True)
#         # 用两个效果好 容易调
#         self.encoder = AVWDCRNN(self.num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers)
#
#         # predictor
#         self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
#
#         self.dropout = nn.Dropout(dropout)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#             else:
#                 nn.init.uniform_(p)
#
#     def forward(self, x):
#         # x: B N T F
#         x = x[..., [0]]
#         source = x.transpose(1, 2).contiguous()
#
#         # source: B, T_1, N, D
#         init_state = self.encoder.init_hidden(source.shape[0])
#         output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
#         output = self.dropout(output)
#         output = output[:, -1:, :, :]  # B, 1, N, hidden
#
#         # CNN based predictor
#         output = self.end_conv(output)  # B, T*C, N, 1
#         output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_nodes)
#         output = output.permute(0, 3, 1, 2).contiguous()  # B, N, T, F
#         return output
#         # return output + x.mean(dim=2, keepdim=True)


import torch
import torch.nn as nn
import torch.nn.functional as F


class AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    def __init__(self, edge_index: 'edge_index', input_dim, rnn_units, output_dim, horizon,
                 num_layers, embed_dim, cheb_k):
        super(AGCRN, self).__init__()

        self.num_node = edge_index.max() + 1
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.name = 'AGCRN-D8'

        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_node, input_dim, rnn_units, cheb_k,
                                embed_dim, num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

        # self.ln = nn.Linear(1, 3)

    def forward(self, x):
        # x: B N T F
        x = x[..., [0]]
        source = x.transpose(1, 2).contiguous()

        # source: B, T_1, N, D

        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv(output)  # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 3, 1, 2).contiguous()  # B, N, T, F

        # output = output.squeeze(dim=-1)

        return output
