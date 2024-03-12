# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.sparse.linalg import eigs


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


class Spatial_Attention_layer(nn.Module):
    """
    compute spatial attention scores
    """

    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()

        self.W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self.W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self.W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self.Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        """

        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)

        rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)

        product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)

        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)

        S_normalized = F.softmax(S, dim=1)

        return S_normalized


class cheb_conv_withSAt(nn.Module):
    """
    K-order chebyshev graph convolution
    """

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        """
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        """
        nn.Module.__init__(self)

        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation
        :param spatial_attention:
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        """

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        # print(len(self.Theta))

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels, device=x.device)  # (b, N, F_out)

            for k in range(self.K):
                T_k = self.cheb_polynomials[k, :, :]  # (N,N)

                T_k_with_at = T_k.mul(spatial_attention)  # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k, :, :]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(
                    graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return F.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        """
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)

        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)

        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)

        E_normalized = F.softmax(E, dim=1)

        return E_normalized


#
#
# class cheb_conv(nn.Module):
#     '''
#     K-order chebyshev graph convolution
#     '''
#
#     def __init__(self, K, cheb_polynomials, in_channels, out_channels):
#         '''
#         :param K: int
#         :param in_channles: int, num of channels in the input sequence
#         :param out_channels: int, num of channels in the output sequence
#         '''
#         super(cheb_conv, self).__init__()
#         self.K = K
#         self.cheb_polynomials = cheb_polynomials
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.Theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
#
#     def forward(self, x):
#         '''
#         Chebyshev graph convolution operation
#         :param x: (batch_size, N, F_in, T)
#         :return: (batch_size, N, F_out, T)
#         '''
#
#         batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
#
#         outputs = []
#
#         for time_step in range(num_of_timesteps):
#
#             graph_signal = x[:, :, :, time_step]  # (b, N, F_in)
#
#             output = torch.zeros(batch_size, num_of_vertices, self.out_channels)  # (b, N, F_out)
#
#             for k in range(self.K):
#                 T_k = self.cheb_polynomials[k, :, :]  # (N,N)
#
#                 theta_k = self.Theta[k, :, :]  # (in_channel, out_channel)
#
#                 rhs = graph_signal.permute(0, 2, 1).matmul(T_k).permute(0, 2, 1)
#
#                 output = output + rhs.matmul(theta_k)
#
#             outputs.append(output.unsqueeze(-1))
#
#         return F.relu(torch.cat(outputs, dim=-1))
#

class ASTGCN_block(nn.Module):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices,
                 num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.TAt = Temporal_Attention_layer(in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        """
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        """
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,
                                                                                               num_of_vertices,
                                                                                               num_of_features,
                                                                                               num_of_timesteps)

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(
            spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual


class ASTGCN_submodule(nn.Module):

    def __init__(self, nb_block, in_channels, out_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCN_submodule, self).__init__()

        self.name = 'ASTGCN'

        self.cheb_polynomials = nn.Parameter(cheb_polynomials, requires_grad=False)

        self.BlockList = nn.ModuleList([ASTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                                                     self.cheb_polynomials, num_of_vertices, len_input)])

        self.BlockList.extend([ASTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, self.cheb_polynomials,
                                            num_of_vertices, len_input // time_strides) for _ in range(nb_block - 1)])

        self.final_conv = nn.Conv2d(int(len_input / time_strides), num_for_predict * out_channels,
                                    kernel_size=(1, nb_time_filter))

        self.out_channels = out_channels

    def forward(self, x):
        x = x[..., [0]]
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        time_len = x.shape[2]
        fea_num = x.shape[3]

        # x: B N T F
        x = x.transpose(2, 3).contiguous()

        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        for block in self.BlockList:
            x = block(x)

        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output.contiguous().view(batch_size, num_nodes, -1, self.out_channels)


class ASTGCN(nn.Module):

    def __init__(self, nb_block, in_channels, out_channels, k, nb_chev_filter, nb_time_filter, time_strides,
                 num_for_predict, len_input, edge_index: 'edge_index', edge_weight: 'edge_weights'):
        '''

        :param DEVICE:
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        :param len_input
        :return:
        '''

        nn.Module.__init__(self)

        self.name = 'ASTGCN'

        num_nodes = edge_index.max().item() + 1
        adj_mx = torch.sparse_coo_tensor(edge_index,
                                         edge_weight, torch.Size([num_nodes, num_nodes])).to_dense().numpy()

        L_tilde = scaled_Laplacian(adj_mx)
        cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor) for i in cheb_polynomial(L_tilde, k)]
        cheb_polynomials = torch.stack(cheb_polynomials, dim=0)
        self.real_model = ASTGCN_submodule(nb_block, in_channels, out_channels, k, nb_chev_filter, nb_time_filter,
                                           time_strides, cheb_polynomials, num_for_predict, len_input, num_nodes)

        for p in self.real_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, *args, **kwargs):
        return self.real_model.forward(*args, **kwargs)