# -*-coding:utf-8-*-
import warnings
from typing import Optional, Sequence, Tuple, Union, List

from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from mmpose.models.heads.base_head import BaseHead
from mmcv.cnn import ConvModule
from torch.nn import functional as F
# from MyWorkDir.old_code.OPECNET.engineer.models.registry import BACKBONES
from MyWorkDir.OPECNET.engineer.models.common.semgcn_helper import _GraphConv, _ResGraphConv_Attention, _ResGraphConv
# from MyWorkDir.old_code.OPECNET.engineer.models.common.HM import HM_Extrect
from scipy import sparse as sp
import numpy as np
from torch import nn
import torch
from MyWorkDir.OPECNET.JudgeModel.GCNModel import JudgeModel

OptIntSeq = Optional[Sequence[int]]


# RTMCCHead
@MODELS.register_module()
class MyHead(BaseHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
            self,
            in_channels: Union[int, Sequence[int]],
            out_channels: int,
            input_size: Tuple[int, int],
            in_featuremap_size: Tuple[int, int],
            simcc_split_ratio: float = 2.0,
            final_layer_kernel_size: int = 1,
            gau_cfg: ConfigType = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            myHead_cfg: ConfigType = dict(
                multihead_hidden_dims=512,
                branch_out=[5, 4, 4, 4, 4],
            ),
            loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
            decoder: OptConfigType = None,
            init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        ### 以下是自己加的
        branch_out = myHead_cfg['branch_out']
        mid_channels = myHead_cfg['multihead_hidden_dims']
        self.branches = []
        for i in range(len(branch_out)):
            stage = []
            stage.append(ConvModule(
                in_channels,
                mid_channels,
                3,
                stride=1,
                padding=1))
            stage.append(nn.Conv2d(
                mid_channels,
                branch_out[i],
                kernel_size=final_layer_kernel_size,
                stride=1,
                padding=final_layer_kernel_size // 2))
            self.add_module(f'branch{i + 1}', nn.Sequential(*stage))
            self.branches.append(f'branch{i + 1}')
        ### 以上是自己加的

        # self.final_layer = nn.Conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size=final_layer_kernel_size,
        #     stride=1,
        #     padding=final_layer_kernel_size // 2)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]
        outs = []
        for i, branch_name in enumerate(self.branches):
            branch = getattr(self, branch_name)
            outs.append(branch(feats))
        feats = torch.cat(tuple(outs), dim=1)
        # feats = self.final_layer(feats)  # -> B, K, H, W

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)  # -> B, K, 64
        feats = self.mlp(feats)  # -> B, K, hidden

        feats = self.gau(feats)

        pred_x = self.cls_x(feats)
        pred_y = self.cls_y(feats)

        return pred_x, pred_y

    def predict(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y = self.forward(_feats)

            _batch_pred_x_flip, _batch_pred_y_flip = self.forward(_feats_flip)
            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
            dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
            dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        # calculate losses
        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        losses.update(loss_kpt=loss)

        # calculate accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )

        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg


# RTMCCHead
@MODELS.register_module()
class MyHeadGCN(BaseHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
            self,
            in_channels: Union[int, Sequence[int]],
            out_channels: int,
            input_size: Tuple[int, int],
            in_featuremap_size: Tuple[int, int],
            simcc_split_ratio: float = 2.0,
            final_layer_kernel_size: int = 1,
            gau_cfg: ConfigType = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            myHead_cfg: ConfigType = dict(
                multihead_hidden_dims=512,
                branch_out=[5, 4, 4, 4, 4],
                gcn_out_nums=3,
                gcn_hidden_dims=256,
            ),
            loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
            decoder: OptConfigType = None,
            init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        ### 以下是自己加的
        branch_out = myHead_cfg['branch_out']
        mid_channels = myHead_cfg['multihead_hidden_dims']
        self.branches = []
        for i in range(len(branch_out)):
            stage = []
            stage.append(ConvModule(
                in_channels,
                mid_channels,
                3,
                stride=1,
                padding=1))
            stage.append(nn.Conv2d(
                mid_channels,
                branch_out[i],
                kernel_size=final_layer_kernel_size,
                stride=1,
                padding=final_layer_kernel_size // 2))
            self.add_module(f'branch{i + 1}', nn.Sequential(*stage))
            self.branches.append(f'branch{i + 1}')
        self.adj = self._build_adj_mx_from_edges()
        # self.gcn_preLayers = _GraphConv(self.adj, 64, 256, p_dropout=None).to(device='cuda:0')
        self.gcn_preLayers = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, myHead_cfg['gcn_hidden_dims'], bias=False),
            ScaleNorm(myHead_cfg['gcn_hidden_dims']))  # 跑了四五个epoch，发现加上这一层scaleNorm时训练更稳定
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        self.out_num = myHead_cfg['gcn_out_nums']
        self.gcn_layers = []
        self.mlps = []
        self.gaus = []
        self.cls_xs = []
        self.cls_ys = []
        for i in range(self.out_num):
            gcn_layer = _ResGraphConv(self.adj, myHead_cfg['gcn_hidden_dims'], myHead_cfg['gcn_hidden_dims'],
                                      myHead_cfg['gcn_hidden_dims'],
                                      p_dropout=None).to(device='cuda:0')
            mlp = nn.Sequential(
                ScaleNorm(myHead_cfg['gcn_hidden_dims']),
                nn.Linear(gau_cfg['hidden_dims'], gau_cfg['hidden_dims'], bias=False))
            gau = RTMCCBlock(
                self.out_channels,
                gau_cfg['hidden_dims'],
                gau_cfg['hidden_dims'],
                s=gau_cfg['s'],
                expansion_factor=gau_cfg['expansion_factor'],
                dropout_rate=gau_cfg['dropout_rate'],
                drop_path=gau_cfg['drop_path'],
                attn_type='self-attn',
                act_fn=gau_cfg['act_fn'],
                use_rel_bias=gau_cfg['use_rel_bias'],
                pos_enc=gau_cfg['pos_enc'])
            cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
            cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
            self.add_module(f'gcn_layer{i + 1}', gcn_layer)
            self.add_module(f'mlp{i + 1}', mlp)
            self.add_module(f'gau{i + 1}', gau)
            self.add_module(f'cls_x{i + 1}', cls_x)
            self.add_module(f'cls_y{i + 1}', cls_y)
            self.gcn_layers.append(f'gcn_layer{i + 1}')
            self.mlps.append(f'mlp{i + 1}')
            self.gaus.append(f'gau{i + 1}')
            self.cls_xs.append(f'cls_x{i + 1}')
            self.cls_ys.append(f'cls_y{i + 1}')

    def _build_adj_mx_from_edges(self):
        edge = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15],
                [15, 16], [17, 18], [18, 19], [19, 20], [1, 5], [5, 9], [9, 13], [13, 17], [0, 1],
                # [0, 5], [0, 9], [0, 13], [0, 17]
                ]
        num_joints = 21

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

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]
        outs = []
        for i, branch_name in enumerate(self.branches):
            branch = getattr(self, branch_name)
            outs.append(branch(feats))
        feats = torch.cat(tuple(outs), dim=1)

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)  # -> B, K, 64
        feats = self.gcn_preLayers(feats)

        last = feats
        pred_xs = []
        pred_ys = []
        for (gcn_layer, mlp, gau, cls_x, cls_y) in zip(self.gcn_layers, self.mlps, self.gaus, self.cls_xs, self.cls_ys):
            gcn_layer = getattr(self, gcn_layer)
            mlp = getattr(self, mlp)
            gau = getattr(self, gau)
            cls_x = getattr(self, cls_x)
            cls_y = getattr(self, cls_y)
            last = gcn_layer(last)
            out = mlp(last)
            out = gau(out)
            pred_x = cls_x(out)
            pred_y = cls_y(out)
            pred_xs.append(pred_x)
            pred_ys.append(pred_y)
            # last = out
        return pred_xs, pred_ys

    def predict(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x_list, _batch_pred_y_list = self.forward(_feats)
            _batch_pred_x_flip_list, _batch_pred_y_flip_list = self.forward(_feats_flip)
            out_len = len(_batch_pred_x_list)
            batch_pred_x = []
            batch_pred_y = []
            for i in range(out_len):
                _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                    _batch_pred_x_flip_list[i],
                    _batch_pred_y_flip_list[i],
                    flip_indices=flip_indices)
                batch_pred_x.append((_batch_pred_x_list[i] + _batch_pred_x_flip) * 0.5)
                batch_pred_y.append((_batch_pred_y_list[i] + _batch_pred_y_flip) * 0.5)
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)
        preds = []
        for batch_pred_xy in zip(batch_pred_x, batch_pred_y):
            preds.append(self.decode(batch_pred_xy))
        # TODO 在这里判断应该选择哪个作为最终输出
        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds[-1]

    def loss(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
            dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
            dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )
        losses = dict()
        factor = 1 / 3  # 作为一个平衡
        gt_simcc = (gt_x, gt_y)
        for (i, x, y) in zip([x for x in range(len(pred_x))], pred_x, pred_y):
            pred_simcc = (x, y)
            loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)
            losses[f'loss{i}'] = loss / factor
            # calculate accuracy
            _, avg_acc, _ = simcc_pck_accuracy(
                output=to_numpy(pred_simcc),
                target=to_numpy(gt_simcc),
                simcc_split_ratio=self.simcc_split_ratio,
                mask=to_numpy(keypoint_weights) > 0,
            )
            acc_pose = torch.tensor(avg_acc, device=gt_x.device)
            losses[f'acc_pose{i}'] = acc_pose

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg


# RTMCCHead
@MODELS.register_module()
class MyHeadGCNTemp(BaseHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    large-kernel convolutional layer, a fully-connected layer and a Gated
    Attention Unit to generate 1d representation from low-resolution feature
    maps.

    Args:
        in_channels (int | sequence[int]): Number of channels in the input
            feature map.
        out_channels (int): Number of channels in the output heatmap.
        input_size (tuple): Size of input image in shape [w, h].
        in_featuremap_size (int | sequence[int]): Size of input feature map.
        simcc_split_ratio (float): Split ratio of pixels.
            Default: 2.0.
        final_layer_kernel_size (int): Kernel size of the convolutional layer.
            Default: 1.
        gau_cfg (Config): Config dict for the Gated Attention Unit.
            Default: dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False).
        loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KLDiscretLoss`
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings
    """

    def __init__(
            self,
            in_channels: Union[int, Sequence[int]],
            out_channels: int,
            input_size: Tuple[int, int],
            in_featuremap_size: Tuple[int, int],
            simcc_split_ratio: float = 2.0,
            final_layer_kernel_size: int = 1,
            gau_cfg: ConfigType = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            myHead_cfg: ConfigType = dict(
                multihead_hidden_dims=512,
                branch_out=[5, 4, 4, 4, 4],
                gcn_out_nums=3,
                gcn_hidden_dims=256,
            ),
            loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
            decoder: OptConfigType = None,
            init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # Define SimCC layers
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        ### 以下是自己加的
        branch_out = myHead_cfg['branch_out']
        mid_channels = myHead_cfg['multihead_hidden_dims']
        self.branches = []
        for i in range(len(branch_out)):
            stage = []
            stage.append(ConvModule(
                in_channels,
                mid_channels,
                3,
                stride=1,
                padding=1))
            stage.append(nn.Conv2d(
                mid_channels,
                branch_out[i],
                kernel_size=final_layer_kernel_size,
                stride=1,
                padding=final_layer_kernel_size // 2))
            self.add_module(f'branch{i + 1}', nn.Sequential(*stage))
            self.branches.append(f'branch{i + 1}')
        self.adj = self._build_adj_mx_from_edges()
        # self.gcn_preLayers = _GraphConv(self.adj, 64, 256, p_dropout=None).to(device='cuda:0')
        self.gcn_preLayers = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, myHead_cfg['gcn_hidden_dims'], bias=False),
            ScaleNorm(myHead_cfg['gcn_hidden_dims']))  # 跑了四五个epoch，发现加上这一层scaleNorm时训练更稳定
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)
        self.out_num = myHead_cfg['gcn_out_nums']
        self.gcn_layers = []
        self.mlps = []
        self.gaus = []
        self.cls_xs = []
        self.cls_ys = []
        for i in range(self.out_num):
            gcn_layer = _ResGraphConv(self.adj, myHead_cfg['gcn_hidden_dims'], myHead_cfg['gcn_hidden_dims'],
                                      myHead_cfg['gcn_hidden_dims'],
                                      p_dropout=None).to(device='cuda:0')
            mlp = nn.Sequential(
                ScaleNorm(myHead_cfg['gcn_hidden_dims']),
                nn.Linear(gau_cfg['hidden_dims'], gau_cfg['hidden_dims'], bias=False))
            gau = RTMCCBlock(
                self.out_channels,
                gau_cfg['hidden_dims'],
                gau_cfg['hidden_dims'],
                s=gau_cfg['s'],
                expansion_factor=gau_cfg['expansion_factor'],
                dropout_rate=gau_cfg['dropout_rate'],
                drop_path=gau_cfg['drop_path'],
                attn_type='self-attn',
                act_fn=gau_cfg['act_fn'],
                use_rel_bias=gau_cfg['use_rel_bias'],
                pos_enc=gau_cfg['pos_enc'])
            cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
            cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)
            self.add_module(f'gcn_layer{i + 1}', gcn_layer)
            self.add_module(f'mlp{i + 1}', mlp)
            self.add_module(f'gau{i + 1}', gau)
            self.add_module(f'cls_x{i + 1}', cls_x)
            self.add_module(f'cls_y{i + 1}', cls_y)
            self.gcn_layers.append(f'gcn_layer{i + 1}')
            self.mlps.append(f'mlp{i + 1}')
            self.gaus.append(f'gau{i + 1}')
            self.cls_xs.append(f'cls_x{i + 1}')
            self.cls_ys.append(f'cls_y{i + 1}')

    def _build_adj_mx_from_edges(self):
        edge = [[1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15],
                [15, 16], [17, 18], [18, 19], [19, 20], [1, 5], [5, 9], [9, 13], [13, 17], [0, 1],
                # [0, 5], [0, 9], [0, 13], [0, 17]
                ]
        num_joints = 21

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

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Forward the network.

        The input is the featuremap extracted by backbone and the
        output is the simcc representation.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            pred_x (Tensor): 1d representation of x.
            pred_y (Tensor): 1d representation of y.
        """
        feats = feats[-1]
        outs = []
        for i, branch_name in enumerate(self.branches):
            branch = getattr(self, branch_name)
            outs.append(branch(feats))
        feats = torch.cat(tuple(outs), dim=1)

        # flatten the output heatmap
        feats = torch.flatten(feats, 2)  # -> B, K, 64
        feats = self.gcn_preLayers(feats)

        last = feats
        pred_xs = []
        pred_ys = []
        for (gcn_layer, mlp, gau, cls_x, cls_y) in zip(self.gcn_layers, self.mlps, self.gaus, self.cls_xs, self.cls_ys):
            gcn_layer = getattr(self, gcn_layer)
            mlp = getattr(self, mlp)
            gau = getattr(self, gau)
            cls_x = getattr(self, cls_x)
            cls_y = getattr(self, cls_y)
            last = gcn_layer(last)
            out = mlp(last)
            out = gau(out)
            pred_x = cls_x(out)
            pred_y = cls_y(out)
            pred_xs.append(pred_x)
            pred_ys.append(pred_y)
            # last = out
        return pred_xs, pred_ys

    def predict(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            List[InstanceData]: The pose predictions, each contains
            the following fields:
                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)
                - keypoint_x_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the x direction
                - keypoint_y_labels (np.ndarray, optional): The predicted 1-D
                    intensity distribution in the y direction
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x_list, _batch_pred_y_list = self.forward(_feats)
            _batch_pred_x_flip_list, _batch_pred_y_flip_list = self.forward(_feats_flip)
            out_len = len(_batch_pred_x_list)
            batch_pred_x = []
            batch_pred_y = []
            for i in range(out_len):
                _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                    _batch_pred_x_flip_list[i],
                    _batch_pred_y_flip_list[i],
                    flip_indices=flip_indices)
                batch_pred_x.append((_batch_pred_x_list[i] + _batch_pred_x_flip) * 0.5)
                batch_pred_y.append((_batch_pred_y_list[i] + _batch_pred_y_flip) * 0.5)
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)
        preds = []
        for batch_pred_xy in zip(batch_pred_x, batch_pred_y):
            preds.append(self.decode(batch_pred_xy))
        # 取高的那个作为输出
        out_preds = preds[0]
        for i in range(0, len(preds[0])):
            for j in range(1, len(preds)):
                score = preds[j][i].keypoint_scores.min()
                row_score = out_preds[i].keypoint_scores.min()
                if (score>row_score):
                    out_preds[i] = preds[j][i]
        preds = out_preds

        # TODO 在这里判断应该选择哪个作为最终输出
        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(preds,
                                                      to_numpy(batch_pred_x),
                                                      to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat([
            d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples
        ],
            dim=0)
        gt_y = torch.cat([
            d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples
        ],
            dim=0)
        keypoint_weights = torch.cat(
            [
                d.gt_instance_labels.keypoint_weights
                for d in batch_data_samples
            ],
            dim=0,
        )
        losses = dict()
        factor = 1 / 3  # 作为一个平衡
        gt_simcc = (gt_x, gt_y)
        for (i, x, y) in zip([x for x in range(len(pred_x))], pred_x, pred_y):
            pred_simcc = (x, y)
            loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)
            losses[f'loss{i}'] = loss / factor
            # calculate accuracy
            _, avg_acc, _ = simcc_pck_accuracy(
                output=to_numpy(pred_simcc),
                target=to_numpy(gt_simcc),
                simcc_split_ratio=self.simcc_split_ratio,
                mask=to_numpy(keypoint_weights) > 0,
            )
            acc_pose = torch.tensor(avg_acc, device=gt_x.device)
            losses[f'acc_pose{i}'] = acc_pose

        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg
