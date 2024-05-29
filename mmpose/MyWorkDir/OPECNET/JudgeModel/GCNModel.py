# -*-coding:utf-8-*-
import torch.nn as nn
from .semgcn_helper import _GraphConv
import numpy as np
from scipy import sparse as sp
import torch


class JudgeModel(nn.Module):
    def __init__(self, mode='t'):
        def get_adj():
            a = [1, 5, 9, 13, 17]
            adj = []
            for i in a:
                adj.append([0, i])
                for idx in range(3):
                    adj.append([i + idx, i + idx + 1])
            for i in range(len(a) - 1):
                adj.append([a[i], a[i + 1]])
            return adj

        adj = self._build_adj_mx_from_edges(21, get_adj())
        super(JudgeModel, self).__init__()
        self.Sigmoid = nn.Sigmoid()

        hide_channel = 20
        self.gconv_input1 = _GraphConv(adj, len(mode), hide_channel)
        self.gconv_input2 = _GraphConv(adj, hide_channel, hide_channel)
        self.gconv_input3 = _GraphConv(adj, hide_channel, hide_channel)
        self.liner = nn.Sequential(
            nn.Linear(21 * hide_channel, 210),
            nn.ReLU(),
            nn.Linear(210, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        out = self.gconv_input1(x)
        out = self.gconv_input2(out)
        out = self.gconv_input3(out)
        out = out.reshape(-1)
        out = self.liner(out)
        return self.Sigmoid(out)

    def _build_adj_mx_from_edges(self, num_joints, edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):
            edges = np.array(edges, dtype=np.int32)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            return adj_mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        return adj_mx_from_edges(num_joints, edge, False)
