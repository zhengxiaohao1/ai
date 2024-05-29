# -*-coding:utf-8-*-
import torch
from mmpose.core import keypoint_pck_accuracy

from MyPBN.OPECNET.engineer.models.registry import BACKBONES
from MyPBN.OPECNET.engineer.models.common.helper import *
from MyPBN.OPECNET.engineer.models.common.semgcn_helper import _ResGraphConv_Attention, SemGraphConv, _GraphConv
from MyPBN.OPECNET.engineer.models.common.HM import HM_Extrect
from scipy import sparse as sp
import numpy as np
import math

# @BACKBONES.register_module
from MyPBN.Utils.Utils import nan_cnt


class SemGCN_FC(nn.Module):
    def __init__(self, hid_dim=None, coords_dim=(2, 2), p_dropout=None):
        '''
        :param adj:  adjacency matrix using for
        :param hid_dim:
        :param coords_dim:
        :param num_layers:
        :param nodes_group:
        :param p_dropout:
        '''

        if hid_dim is None:
            hid_dim = [128, 128, 128, 128, 128]
        adj = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15],
               [15, 16], [17, 18], [18, 19], [19, 20]
            , [1, 5], [5, 9], [9, 13], [13, 17]
            , [0, 1],  # [0, 5], [0, 9], [0, 13], [0, 17]
               ]
        num_joints = 21
        super(SemGCN_FC, self).__init__()

        self.heat_map_head = []

        self.gcn_head = []
        self.generator_map = []
        self.adj = self._build_adj_mx_from_edges(num_joints, adj)
        adj = self.adj_matrix

        fea_channels = [256, 256]
        # fea_channels = [280, 640]
        fea = 128
        self.gconv_input = _GraphConv(adj, coords_dim[0], fea, p_dropout=p_dropout).to(device='cuda:0')
        # in here we set 4 gcn model in this par).to(device='cuda:0)
        self.gconv_layers1 = _ResGraphConv_Attention(adj, fea, fea, fea, p_dropout=p_dropout).to(device='cuda:0')
        self.gconv_layers2 = _ResGraphConv_Attention(adj, fea + fea_channels[0], fea + fea_channels[0],
                                                     fea + fea_channels[0], p_dropout=p_dropout).to(device='cuda:0')
        self.gconv_layers3 = _ResGraphConv_Attention(adj, fea + fea_channels[0] + fea_channels[1],
                                                     fea + fea_channels[0] + fea_channels[1],
                                                     fea + fea_channels[0] + fea_channels[1], p_dropout=p_dropout).to(
            device='cuda:0')
        # self.gconv_layers4 = _ResGraphConv_Attention(adj, 1038, 1038, 1038, p_dropout=p_dropout).to(device='cuda:0')

        self.gconv_output1 = SemGraphConv(fea + fea_channels[0], coords_dim[1], adj).to(device='cuda:0')
        self.gconv_output2 = SemGraphConv(fea + fea_channels[0] + fea_channels[1], coords_dim[1], adj).to(
            device='cuda:0')

        self.gcn_head.append(self.gconv_input)
        self.gcn_head.append(self.gconv_layers1)
        self.gcn_head.append(self.gconv_layers2)
        self.gcn_head.append(self.gconv_layers3)
        # self.gcn_head.append(self.gconv_layers4)
        self.gcn_head.append(self.gconv_output1)
        self.gcn_head.append(self.gconv_output2)

        # # FC
        # self.FC = nn.Sequential(nn.Sequential(make_fc(5120, 1024), nn.ReLU(inplace=True)),
        #                         nn.Sequential(make_fc(1024, 1024), nn.ReLU(inplace=True)), make_fc(1024, 2))
        # self.gcn_head.append(self.FC)

    def extract_features_joints(self, ret_features, hms):
        '''
        extract features from joint feature_map

        :return:
        '''

        joint_features = []

        for feature, hm_pred in zip(ret_features, hms):
            joint_feature = torch.zeros([feature.shape[0], feature.shape[1], hm_pred.shape[1]]).cuda()
            for bz in range(feature.shape[0]):
                for joint in range(hm_pred.shape[1]):
                    joint_feature[bz, :, joint] = feature[bz, :, hm_pred[bz, joint, 1], hm_pred[bz, joint, 0]]
            joint_features.append(joint_feature)
        return joint_features

    @property
    def adj_matrix(self):
        return self.adj

    @adj_matrix.setter
    def adj_matrix(self, adj_matrix):
        self.adj = adj_matrix

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

    def forward(self, x, ret_features):
        j1 = ret_features[0]  # 270
        j2 = ret_features[1]  # 640

        print_list = []
        # x = torch.cat([hm_4,score],-1)
        out = self.gconv_input(x)
        print_list.append(nan_cnt(out))
        # gconv_layers in here is residual GCN.
        # label == 0
        out = self.gconv_layers1(out, None)
        print_list.append(nan_cnt(out))
        out = self.gconv_layers2(out, j1)
        print_list.append(nan_cnt(out))
        out1 = self.gconv_output1(out)
        print_list.append(nan_cnt(out))

        out = self.gconv_layers3(out, j2)
        print_list.append(nan_cnt(out))
        # out = self.gconv_layers4(out, None)
        out2 = self.gconv_output2(out)
        print_list.append(nan_cnt(out))
        if nan_cnt(out) > 0:
            print("GCN 第153行")
            print(print_list)
            j1 = j1.detach().cpu().numpy()
            j2 = j2.detach().cpu().numpy()
            print(j1.min())
            print(j1.max())
            print(np.mean(j1))
            print(j2.min())
            print(j2.max())
            print(np.mean(j2))
            exit(-1)
        return [out1, out2]
