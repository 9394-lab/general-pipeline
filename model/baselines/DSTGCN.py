import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
from torch import nn

"""
第11篇引文的复现 DYNAMIC SPATIOTEMPORAL GRAPH CONVOLUTIONAL NEURAL NETWORKS FOR TRAFFIC DATA IMPUTATION WITH 
COMPLEX MISSING PATTERNS
"""


def compute_chebyshev_polynomials(k, A):
    # compute Chebyshev polynomials up to order k. Return a list of matrices.
    # print(f"Computing Chebyshev polynomials up to order {self.K}.")
    cheby_A = []
    for a in A:
        for i in range(k + 1):
            if i == 0:
                I = torch.eye(a.shape[0], device=a.device)
                cheby_A.append(I)
            elif i == 1:
                cheby_A.append(a)
            else:
                cheby_A.append(2 * torch.mm(a, cheby_A[k - 1]) - cheby_A[k - 2])
    return cheby_A


def asym_adj(adj):  # asymmetric
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    rowsum[rowsum <= 1e-5] = 1e-5
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


class GSE(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, emb_dim):
        nn.Module.__init__(self)
        self.forward_LN1 = nn.Linear(input_channels, hidden_channels, bias=True)
        self.forward_LN2 = nn.Linear(hidden_channels, output_channels, bias=True)
        self.backward_LN1 = nn.Linear(input_channels, hidden_channels, bias=True)
        self.backward_LN2 = nn.Linear(hidden_channels, output_channels, bias=True)
        self.RelU1 = nn.ReLU()
        self.RelU2 = nn.ReLU()
        self.forward_W1 = nn.Parameter(torch.FloatTensor(output_channels, emb_dim))
        self.forward_W2 = nn.Parameter(torch.FloatTensor(output_channels, emb_dim))
        self.forward_b = nn.Parameter(torch.FloatTensor(output_channels))
        self.backward_W1 = nn.Parameter(torch.FloatTensor(output_channels, emb_dim))
        self.backward_W2 = nn.Parameter(torch.FloatTensor(output_channels, emb_dim))
        self.backward_b = nn.Parameter(torch.FloatTensor(output_channels))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, for_x, back_x, Adj):
        for_x = self.forward_LN1(for_x)
        for_x = self.RelU1(for_x)
        for_Dyadj = self.forward_LN2(for_x)
        back_x = self.backward_LN1(back_x)
        back_x = self.RelU2(back_x)
        back_Dyadj = self.backward_LN2(back_x)
        for_adj, back_adj = Adj
        for_gate = torch.einsum('ij,im->ij', [for_Dyadj, self.forward_W1]) + \
                   torch.einsum('ij,im->ij', [for_adj, self.forward_W2]) + self.forward_b
        back_gate = torch.einsum('ij,im->ij', [back_Dyadj, self.forward_W1]) + \
                    torch.einsum('ij,im->ij', [back_adj, self.forward_W2]) + self.backward_b
        for_gate = torch.sigmoid(for_gate)
        back_gate = torch.sigmoid(back_gate)
        for_Adj = for_gate * for_adj + (1 - for_gate) * for_Dyadj
        back_Adj = back_gate * back_adj + (1 - back_gate) * back_Dyadj
        return for_Adj, back_Adj


class GCN(nn.Module):
    def __init__(self, K: int, input_dim: int, hidden_dim: int, bias=True, activation=nn.ReLU):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=(K+1)*2)

    def init_params(self, n_supports: int, b_init=0):
        self.W = nn.Parameter(torch.empty(n_supports * self.input_dim, self.hidden_dim), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, A: torch.Tensor, x: torch.Tensor):
        """
        Batch-wise graph convolution operation on given list of support adj matrices
        :param A: support adj matrices - torch.Tensor len:K*2 (n_nodes, n_nodes)
        :param x: graph feature/signal - torch.Tensor (batch_size, n_nodes, time_len, input_dim)
        :return: hidden representation - torch.Tensor (batch_size, n_nodes, time_len, hidden_dim)
        """
        support_list = list()
        for a in A:
            support = torch.einsum('mn,bntf->bmtf', [a, x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum('bntf,fy->bnty', [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return self.__class__.__name__ + f'({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})'


class DSTGCN(nn.Module):
    def __init__(self, edge_index: 'edge_index', edge_weight: 'edge_weights', input_channels=3, output_channels=64,
                 input_time_len=12, output_time_len=1, lstm_layer=4, k=2, seen_ratio=0.9):
        nn.Module.__init__(self)
        self.name = 'DSTGCN-D3'
        self.output_channels = output_channels
        self.k = k
        self.LSTM = nn.LSTM(input_size=input_channels, hidden_size=output_channels, num_layers=lstm_layer,
                            batch_first=True, bidirectional=True)
        self.LN1 = nn.Linear(output_channels*2, output_channels)

        num_nodes = edge_index.max().item() + 1
        supports = torch.sparse_coo_tensor(edge_index, edge_weight, [num_nodes, num_nodes]).to_dense().numpy()
        supports = [asym_adj(supports), asym_adj(np.transpose(supports))]
        self.Af = nn.Parameter(torch.tensor(supports[0]), requires_grad=False)
        self.Ab = nn.Parameter(torch.tensor(supports[1]), requires_grad=False)
        self.Adj = [self.Af, self.Ab]

        self.GSE = GSE(output_channels, output_channels, num_nodes, 10)
        self.GCN = GCN(k, output_channels, output_channels)

        self.Ln = nn.Sequential(nn.Linear(output_channels, output_channels),
                                nn.ReLU(), )
        self.endconv = nn.Linear(input_time_len * output_channels, output_time_len)

    def forward(self, x):
        # input: B N T F
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]

        x = x.view(batch_size * num_nodes, time_len, -1)

        LSTM_out, _ = self.LSTM(x.view(batch_size * num_nodes, time_len, -1))
        LSTM_out = LSTM_out.contiguous().view(batch_size, num_nodes, time_len, -1)
        hft, hbt = LSTM_out[..., :self.output_channels], LSTM_out[..., self.output_channels:]
        MLSTM = self.LN1(LSTM_out)
        for_x, back_x = hft[-1, :, -1, :].squeeze(), hbt[-1, :, -1, :].squeeze()
        Dynamic_Adj = self.GSE(for_x, back_x, self.Adj)
        Dynamic_Adj = compute_chebyshev_polynomials(self.k, Dynamic_Adj)
        MGCN = self.GCN(Dynamic_Adj, MLSTM)
        Output = F.layer_norm(MLSTM + MGCN, [self.output_channels])
        Output = self.Ln(Output).contiguous().view(batch_size, num_nodes, -1)
        Output = self.endconv(Output)
        return Output.unsqueeze(-1)

