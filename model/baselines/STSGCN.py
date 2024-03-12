import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs

#
# class gcn_operation(nn.Module):
#     def __init__(self, adj, in_dim, out_dim, num_vertices, activation='GLU'):
#         """
#         图卷积模块
#         :param adj: 邻接图
#         :param in_dim: 输入维度
#         :param out_dim: 输出维度
#         :param num_vertices: 节点数量
#         :param activation: 激活方式 {'relu', 'GLU'}
#         """
#         super(gcn_operation, self).__init__()
#         self.adj = adj
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_vertices = num_vertices
#         self.activation = activation
#
#         assert self.activation in {'GLU', 'relu'}
#
#         if self.activation == 'GLU':
#             self.FC = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
#         else:
#             self.FC = nn.Linear(self.in_dim, self.out_dim, bias=True)
#
#     def forward(self, x, mask=None):
#         """
#         :param x: (3*N, B, Cin)
#         :param mask:(3*N, 3*N)
#         :return: (3*N, B, Cout)
#         """
#         adj = self.adj
#         if mask is not None:
#             adj = adj.to(mask.device) * mask
#
#         x = torch.einsum('nm, mbc->nbc', adj.to(x.device, torch.float32), x)  # 3*N, B, Cin
#
#         if self.activation == 'GLU':
#             lhs_rhs = self.FC(x)  # 3*N, B, 2*Cout
#             lhs, rhs = torch.split(lhs_rhs, self.out_dim, dim=-1)  # 3*N, B, Cout
#
#             out = lhs * torch.sigmoid(rhs)
#             del lhs, rhs, lhs_rhs
#
#             return out
#
#         elif self.activation == 'relu':
#             return torch.relu(self.FC(x))  # 3*N, B, Cout
#
#
# class STSGCM(nn.Module):
#     def __init__(self, adj, in_dim, out_dims, num_of_vertices, activation='GLU'):
#         """
#         :param adj: 邻接矩阵
#         :param in_dim: 输入维度
#         :param out_dims: list 各个图卷积的输出维度
#         :param num_of_vertices: 节点数量
#         :param activation: 激活方式 {'relu', 'GLU'}
#         """
#         super(STSGCM, self).__init__()
#         self.adj = adj
#         self.in_dim = in_dim
#         self.out_dims = out_dims
#         self.num_of_vertices = num_of_vertices
#         self.activation = activation
#
#         self.gcn_operations = nn.ModuleList()
#
#         self.gcn_operations.append(
#             gcn_operation(
#                 adj=self.adj,
#                 in_dim=self.in_dim,
#                 out_dim=self.out_dims[0],
#                 num_vertices=self.num_of_vertices,
#                 activation=self.activation
#             )
#         )
#
#         for i in range(1, len(self.out_dims)):
#             self.gcn_operations.append(
#                 gcn_operation(
#                     adj=self.adj,
#                     in_dim=self.out_dims[i-1],
#                     out_dim=self.out_dims[i],
#                     num_vertices=self.num_of_vertices,
#                     activation=self.activation
#                 )
#             )
#
#     def forward(self, x, mask=None):
#         """
#         :param x: (3N, B, Cin)
#         :param mask: (3N, 3N)
#         :return: (N, B, Cout)
#         """
#         need_concat = []
#
#         for i in range(len(self.out_dims)):
#             x = self.gcn_operations[i](x, mask)
#             need_concat.append(x)
#
#         # shape of each element is (1, N, B, Cout)
#         need_concat = [
#             torch.unsqueeze(
#                 h[self.num_of_vertices: 2 * self.num_of_vertices], dim=0
#             ) for h in need_concat
#         ]
#
#         out = torch.max(torch.cat(need_concat, dim=0), dim=0).values  # (N, B, Cout)
#
#         del need_concat
#
#         return out
#
#
# class STSGCL(nn.Module):
#     def __init__(self,
#                  adj,
#                  history,
#                  num_of_vertices,
#                  in_dim,
#                  out_dims,
#                  strides=3,
#                  activation='GLU',
#                  temporal_emb=True,
#                  spatial_emb=True):
#         """
#         :param adj: 邻接矩阵
#         :param history: 输入时间步长
#         :param in_dim: 输入维度
#         :param out_dims: list 各个图卷积的输出维度
#         :param strides: 滑动窗口步长，local时空图使用几个时间步构建的，默认为3
#         :param num_of_vertices: 节点数量
#         :param activation: 激活方式 {'relu', 'GLU'}
#         :param temporal_emb: 加入时间位置嵌入向量
#         :param spatial_emb: 加入空间位置嵌入向量
#         """
#         super(STSGCL, self).__init__()
#         self.adj = adj
#         self.strides = strides
#         self.history = history
#         self.in_dim = in_dim
#         self.out_dims = out_dims
#         self.num_of_vertices = num_of_vertices
#
#         self.activation = activation
#         self.temporal_emb = temporal_emb
#         self.spatial_emb = spatial_emb
#
#         self.STSGCMS = nn.ModuleList()
#         for i in range(self.history - self.strides + 1):
#             self.STSGCMS.append(
#                 STSGCM(
#                     adj=self.adj,
#                     in_dim=self.in_dim,
#                     out_dims=self.out_dims,
#                     num_of_vertices=self.num_of_vertices,
#                     activation=self.activation
#                 )
#             )
#
#         if self.temporal_emb:
#             self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.history, 1, self.in_dim))
#             # 1, T, 1, Cin
#
#         if self.spatial_emb:
#             self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, 1, self.num_of_vertices, self.in_dim))
#             # 1, 1, N, Cin
#
#         self.reset()
#
#     def reset(self):
#         if self.temporal_emb:
#             nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)
#
#         if self.spatial_emb:
#             nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)
#
#     def forward(self, x, mask=None):
#         """
#         :param x: B, T, N, Cin
#         :param mask: (N, N)
#         :return: B, T-2, N, Cout
#         """
#         if self.temporal_emb:
#             x = x + self.temporal_embedding
#
#         if self.spatial_emb:
#             x = x + self.spatial_embedding
#
#         need_concat = []
#         batch_size = x.shape[0]
#
#         for i in range(self.history - self.strides + 1):
#             t = x[:, i: i+self.strides, :, :]  # (B, 3, N, Cin)
#
#             t = torch.reshape(t, shape=[batch_size, self.strides * self.num_of_vertices, self.in_dim])
#             # (B, 3*N, Cin)
#
#             t = self.STSGCMS[i](t.permute(1, 0, 2), mask)  # (3*N, B, Cin) -> (N, B, Cout)
#
#             t = torch.unsqueeze(t.permute(1, 0, 2), dim=1)  # (N, B, Cout) -> (B, N, Cout) ->(B, 1, N, Cout)
#
#             need_concat.append(t)
#
#         out = torch.cat(need_concat, dim=1)  # (B, T-2, N, Cout)
#
#         del need_concat, batch_size
#
#         return out
#
#
# class output_layer(nn.Module):
#     def __init__(self, num_of_vertices, history, in_dim,
#                  hidden_dim=128, horizon=12):
#         """
#         预测层，注意在作者的实验中是对每一个预测时间step做处理的，也即他会令horizon=1
#         :param num_of_vertices:节点数
#         :param history:输入时间步长
#         :param in_dim: 输入维度
#         :param hidden_dim:中间层维度
#         :param horizon:预测时间步长
#         """
#         super(output_layer, self).__init__()
#         self.num_of_vertices = num_of_vertices
#         self.history = history
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.horizon = horizon
#
#         self.FC1 = nn.Linear(self.in_dim * self.history, self.hidden_dim, bias=True)
#
#         self.FC2 = nn.Linear(self.hidden_dim, self.horizon, bias=True)
#
#     def forward(self, x):
#         """
#         :param x: (B, Tin, N, Cin)
#         :return: (B, Tout, N)
#         """
#         batch_size = x.shape[0]
#
#         x = x.permute(0, 2, 1, 3)  # B, N, Tin, Cin
#
#         out1 = torch.relu(self.FC1(x.reshape(batch_size, self.num_of_vertices, -1)))
#         # (B, N, Tin, Cin) -> (B, N, Tin * Cin) -> (B, N, hidden)
#
#         out2 = self.FC2(out1)  # (B, N, hidden) -> (B, N, horizon)
#
#         del out1, batch_size
#
#         return out2.permute(0, 2, 1)  # B, horizon, N
#
#
# class STSGCN(nn.Module):
#     def __init__(self, edge_index: 'edge_index', edge_weight: 'edge_weights', history, in_dim,
#                  hidden_dims, first_layer_embedding_size, out_layer_dim, activation='GLU', use_mask=True,
#                  temporal_emb=True, spatial_emb=True, horizon=12, strides=3):
#         """
#         :param edge_index:
#         :param edge_weight:
#         :param history:输入时间步长
#         :param in_dim:输入维度
#         :param hidden_dims: lists, 中间各STSGCL层的卷积操作维度
#         :param first_layer_embedding_size: 第一层输入层的维度
#         :param out_layer_dim: 输出模块中间层维度
#         :param activation: 激活函数 {relu, GlU}
#         :param use_mask: 是否使用mask矩阵对adj进行优化
#         :param temporal_emb:是否使用时间嵌入向量
#         :param spatial_emb:是否使用空间嵌入向量
#         :param horizon:预测时间步长
#         :param strides:滑动窗口步长，local时空图使用几个时间步构建的，默认为3
#         """
#         super(STSGCN, self).__init__()
#         self.name = 'STSGCN'
#         self.adj = construct_adj(edge_index, edge_weight, strides)
#         self.num_of_vertices = edge_index.max().item() + 1
#         self.hidden_dims = hidden_dims
#         self.out_layer_dim = out_layer_dim
#         self.activation = activation
#         self.use_mask = use_mask
#
#         self.temporal_emb = temporal_emb
#         self.spatial_emb = spatial_emb
#         self.horizon = horizon
#         self.strides = strides
#
#         self.First_FC = nn.Linear(in_dim, first_layer_embedding_size, bias=True)
#         self.STSGCLS = nn.ModuleList()
#         self.STSGCLS.append(
#             STSGCL(
#                 adj=self.adj,
#                 history=history,
#                 num_of_vertices=self.num_of_vertices,
#                 in_dim=first_layer_embedding_size,
#                 out_dims=self.hidden_dims[0],
#                 strides=self.strides,
#                 activation=self.activation,
#                 temporal_emb=self.temporal_emb,
#                 spatial_emb=self.spatial_emb
#             )
#         )
#
#         in_dim = self.hidden_dims[0][-1]
#         history -= (self.strides - 1)
#
#         for idx, hidden_list in enumerate(self.hidden_dims):
#             if idx == 0:
#                 continue
#             self.STSGCLS.append(
#                 STSGCL(
#                     adj=self.adj,
#                     history=history,
#                     num_of_vertices=self.num_of_vertices,
#                     in_dim=in_dim,
#                     out_dims=hidden_list,
#                     strides=self.strides,
#                     activation=self.activation,
#                     temporal_emb=self.temporal_emb,
#                     spatial_emb=self.spatial_emb
#                 )
#             )
#
#             history -= (self.strides - 1)
#             in_dim = hidden_list[-1]
#
#         self.predictLayer = nn.ModuleList()
#         for t in range(self.horizon):
#             self.predictLayer.append(
#                 output_layer(
#                     num_of_vertices=self.num_of_vertices,
#                     history=history,
#                     in_dim=in_dim,
#                     hidden_dim=out_layer_dim,
#                     horizon=1
#                 )
#             )
#
#         if self.use_mask:
#             mask = torch.zeros_like(self.adj)
#             mask[self.adj != 0] = self.adj[self.adj != 0]
#             self.mask = nn.Parameter(mask)
#         else:
#             self.mask = None
#
#     def forward(self, x):
#         """
#         :param x: B, N, Tin, Cin
#         :return: B, Tout, N
#         """
#         x = x.permute(0, 2, 1, 3)
#         x = torch.relu(self.First_FC(x))  # B, Tin, N, Cin
#
#         for model in self.STSGCLS:
#             x = model(x, self.mask)
#         # (B, T - 8, N, Cout)
#
#         need_concat = []
#         for i in range(self.horizon):
#             out_step = self.predictLayer[i](x)  # (B, 1, N)
#             need_concat.append(out_step)
#
#         out = torch.cat(need_concat, dim=1)  # B, Tout, N
#
#         del need_concat
#
#         return out.permute(0, 2, 1).unsqueeze(-1)


def scaled_Laplacian(W):
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def construct_adj(edge_index, edge_weight, steps):
    """
    构建local 时空图
    :param edge_weight:
    :param edge_index:
    :param steps: 选择几个时间步来构建图
    :return: new adjacency matrix: csr_matrix, shape is (N * steps, N * steps)
    """
    N = edge_index.max().item() + 1
    A = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([N, N])).to_dense().numpy()
    adj = np.zeros((N * steps, N * steps))

    for i in range(steps):
        """对角线代表各个时间步自己的空间图，也就是A"""
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            """每个节点只会连接相邻时间步的自己"""
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        """加入自回"""
        adj[i, i] = 1

    return torch.tensor(adj, dtype=torch.float32)


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, A, x):
        x = torch.einsum('vn,bfnt->bfvt', (A, x))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn_glu(nn.Module):
    def __init__(self, c_in, c_out):
        super(gcn_glu, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, 2 * c_out)
        self.c_out = c_out

    def forward(self, x, A):
        # (3N, B, C)
        x = x.unsqueeze(3)  # (3N, B, C, 1)
        x = x.permute(1, 2, 0, 3)  # (3N, B, C, 1)->(B, C, 3N, 1)
        ax = self.nconv(A, x)
        axw = self.mlp(ax)  # (B, 2C', 3N, 1)
        axw_1, axw_2 = torch.split(axw, [self.c_out, self.c_out], dim=1)
        axw_new = axw_1 * torch.sigmoid(axw_2)  # (B, C', 3N, 1)
        axw_new = axw_new.squeeze(3)  # (B, C', 3N)
        axw_new = axw_new.permute(2, 0, 1)  # (3N, B, C')
        return axw_new


class stsgcm(nn.Module):
    def __init__(self, num_nodes, gcn_num, num_of_features, output_features_num):
        super(stsgcm, self).__init__()
        c_in = num_of_features
        c_out = output_features_num
        self.gcn_glu = nn.ModuleList()
        for _ in range(gcn_num):
            self.gcn_glu.append(gcn_glu(c_in, c_out))
            c_in = c_out
        self.num_nodes = num_nodes
        self.gcn_num = gcn_num

    def forward(self, x, A):
        # (3N, B, C)
        need_concat = []
        for i in range(self.gcn_num):
            x = self.gcn_glu[i](x, A)
            need_concat.append(x)
        # (3N, B, C')
        need_concat = [i[(self.num_nodes):(2 * self.num_nodes), :, :].unsqueeze(0) for i in
                       need_concat]  # (1, N, B, C')
        outputs = torch.stack(need_concat, dim=0)  # (3, N, B, C')
        outputs = torch.max(outputs, dim=0).values  # (1, N, B, C')
        return outputs


class stsgcl(nn.Module):
    def __init__(self, num_of_vertices, nhid, gcn_num, input_length, input_features_num):
        super(stsgcl, self).__init__()
        # self.position_embedding = position_embedding(args)
        self.T = input_length
        self.num_of_vertices = num_of_vertices
        self.input_features_num = input_features_num
        output_features_num = nhid
        self.stsgcm = nn.ModuleList()
        for _ in range(self.T - 2):
            self.stsgcm.append(stsgcm(num_of_vertices, gcn_num, self.input_features_num, output_features_num))

        # position_embedding
        self.temporal_emb = torch.nn.init.xavier_normal_(torch.empty(1, self.T, 1, self.input_features_num),
                                                         gain=0.0003)
        self.spatial_emb = torch.nn.init.xavier_normal_(
            torch.empty(1, 1, self.num_of_vertices, self.input_features_num), gain=0.0003)
        # self.temporal_emb = torch.nn.init.xavier_uniform_(torch.empty(1, self.T, 1,self.input_features_num),
        # gain=1).cuda() self.spatial_emb = torch.nn.init.xavier_uniform_(torch.empty(1, 1, self.num_of_vertices,
        # self.input_features_num),gain=1).cuda()

    def forward(self, x, A):
        # (B, T, N, C)
        # position_embedding
        x = x + self.temporal_emb.to(x.device)
        x = x + self.spatial_emb.to(x.device)
        data = x
        need_concat = []
        for i in range(self.T - 2):
            # shape is (B, 3, N, C)
            t = data[:, i:i + 3, :, :]
            # shape is (B, 3N, C)
            t = t.reshape([-1, 3 * self.num_of_vertices, self.input_features_num])
            # shape is (3N, B, C)
            t = t.permute(1, 0, 2)
            # shape is (1, N, B, C')
            t = self.stsgcm[i](t, A)
            # shape is (B, 1, N, C')
            t = t.permute(2, 0, 1, 3).squeeze(1)
            need_concat.append(t)
        outputs = torch.stack(need_concat, dim=1)  # (B, T - 2, N, C')
        return outputs  # (B, T - 2, N, C')


class output_layer(nn.Module):
    def __init__(self, nhid, input_length):
        super(output_layer, self).__init__()
        self.fully_1 = torch.nn.Conv2d(input_length * nhid, 128, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
                                       bias=True)
        self.fully_2 = torch.nn.Conv2d(128, 1, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, data):
        # (B, T, N, C)
        _, time_num, node_num, feature_num = data.size()
        data = data.permute(0, 2, 1, 3)  # (B, T, N, C)->(B, N, T, C)
        data = data.reshape([-1, node_num, time_num * feature_num, 1])  # (B, N, T, C)->(B, N, T*C, 1)
        data = data.permute(0, 2, 1, 3)  # (B, N, T*C, 1)->(B, T*C, N, 1)
        data = self.fully_1(data)  # (B, 128, N, 1)
        data = torch.relu(data)
        data = self.fully_2(data)  # (B, 1, N, 1)
        data = data.squeeze(dim=3)  # (B, 1, N)
        return data  # (B, 1, N)


class STSGCN(nn.Module):
    def __init__(self, edge_index: 'edge_index', edge_weight: 'edge_weights', strides, layer_num, input_length, in_dim,
                 first_layer_embedding_size, horizon, nhid, gcn_num):
        super(STSGCN, self).__init__()
        self.name = 'STSGCN'
        self.A = construct_adj(edge_index, edge_weight, strides)
        num_of_vertices = edge_index.max().item() + 1
        self.layer_num = layer_num
        input_features_num = nhid
        self.predict_length = horizon
        self.mask = nn.Parameter(torch.rand(strides * num_of_vertices, strides * num_of_vertices),
                                 requires_grad=True)
        self.stsgcl = nn.ModuleList()
        for _ in range(self.layer_num):
            self.stsgcl.append(stsgcl(num_of_vertices, nhid, gcn_num, input_length, input_features_num))
            input_length -= 2
        self.output_layer = nn.ModuleList()
        for _ in range(self.predict_length):
            self.output_layer.append(output_layer(nhid, input_length))
        self.input_layer = torch.nn.Conv2d(in_dim, first_layer_embedding_size, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
                                           bias=True)

    def forward(self, input):
        # （B,N,T,C）
        input = input[..., [0]]
        input = input.permute(0, 3, 1, 2)  # （B,C,N,T）
        data = self.input_layer(input)
        data = torch.relu(data)
        data = data.permute(0, 3, 2, 1)  # （B,T,N,C'）
        adj = self.mask * self.A.to(self.mask.device)
        for i in range(self.layer_num):
            data = self.stsgcl[i](data, adj)
        # (B, 4, N, C')
        need_concat = []
        for i in range(self.predict_length):
            output = self.output_layer[i](data)  # (B, 1, N)
            need_concat.append(output.squeeze(1))
        outputs = torch.stack(need_concat, dim=1)  # (B, 12, N)
        outputs = outputs.unsqueeze(3)  # (B, 12, N, 1)
        return outputs.permute(0, 2, 1, 3)
