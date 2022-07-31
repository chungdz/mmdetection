# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import force_fp32

from mmdet.models.builder import ROI_EXTRACTORS
from .base_roi_extractor import BaseRoIExtractor


@ROI_EXTRACTORS.register_module()
class SingleRoIExtractorGeM(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(SingleRoIExtractorGeM, self).__init__(roi_layer, out_channels,
                                                 featmap_strides, init_cfg)
        self.finest_scale = finest_scale
        self.p = nn.Parameter(torch.Tensor([3, 3, 3, 3]))
        self.proj = nn.ModuleList([nn.Conv2d(256, 256, 1, 1, 0) for _ in range(4)])
        self.minimumx = nn.Parameter(torch.Tensor([1e-6]), requires_grad=False)
        # self.weights = nn.Parameter(torch.Tensor([[0.8, 0.4, 0.2, 0.1],
        #                                                 [0.2, 0.8, 0.4, 0.1],
        #                                                 [0.1, 0.2, 0.8, 0.4],
        #                                                 [0.1, 0.2, 0.4, 0.8]]), requires_grad=False)
        self.weights = nn.Parameter(torch.Tensor([[0.8, 0, 0, 0],
                                                        [0.4, 0.8, 0, 0],
                                                        [0.2, 0.4, 0.8, 0],
                                                        [0.1, 0.2, 0.4, 0.8]]), requires_grad=False)
        self.output_size = roi_layer.output_size

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        output_dim = self.output_size

        roi_feats_list = []
        rois_index = rois[:, 0].long()
        rois_index_sub = rois_index.reshape(-1, 1, 1).repeat(1, output_dim, output_dim)
        rlen = len(rois)
        for level in range(num_levels):
            curf = feats[level]
            batch_size, channels, ch, cw = curf.size()
            scale_factor = self.featmap_strides[level]
            rois_rescale = torch.round(rois[:, 1:] / scale_factor).long()
            # gem power
            xpower = torch.pow(torch.maximum(curf, self.minimumx), self.p[level])
            # gem sum, but store as cumsum
            xpower_cumsum = xpower.cumsum(2).cumsum(3)
            # rescaled coordinates
            x1 = torch.clamp(rois_rescale[:, 0], min=1, max=cw - 1)
            y1 = torch.clamp(rois_rescale[:, 1], min=1, max=ch - 1)
            x2 = torch.clamp(rois_rescale[:, 2], min=1, max=cw - 1)
            y2 = torch.clamp(rois_rescale[:, 3], min=1, max=ch - 1)
            # subregion intervals generate output_dim * output_dim
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            winc = torch.floor(w / output_dim).long()
            hinc = torch.floor(h / output_dim).long()
            # find all subregion coordinates and area
            subrois_x1 = []
            subrois_x2 = []
            subrois_y1 = []
            subrois_y2 = []
            subrois_area = []
            for i in range(output_dim):
                cur_y1 = y1 + hinc * i
                if i < 6:
                    cur_y2 = cur_y1 + hinc
                else:
                    cur_y2 = y2
                for j in range(output_dim):
                    cur_x1 = x1 + winc * j
                    if j < 6:
                        cur_x2 = cur_x1 + winc
                    else:
                        cur_x2 = x2

                    subrois_x1.append(cur_x1)
                    subrois_y1.append(cur_y1)
                    subrois_x2.append(cur_x2)
                    subrois_y2.append(cur_y2)
                    
                    subarea = (cur_x2 - cur_x1 + 1) * (cur_y2 - cur_y1 + 1)
                    subrois_area.append(subarea)
            # concat them all together
            # size(N, output_dim, output_dim)
            sx1 = torch.cat(subrois_x1, dim=-1).reshape(output_dim, output_dim, rlen).permute(2, 1, 0)
            sy1 = torch.cat(subrois_y1, dim=-1).reshape(output_dim, output_dim, rlen).permute(2, 1, 0)
            sx2 = torch.cat(subrois_x2, dim=-1).reshape(output_dim, output_dim, rlen).permute(2, 1, 0)
            sy2 = torch.cat(subrois_y2, dim=-1).reshape(output_dim, output_dim, rlen).permute(2, 1, 0)
            ss = torch.cat(subrois_area, dim=-1).reshape(output_dim, output_dim, rlen).permute(2, 1, 0)
            # get ROI subregion gem sum
            v1 = xpower_cumsum[rois_index_sub, :, sy1 - 1, sx1 - 1]
            v2 = xpower_cumsum[rois_index_sub, :, sy1 - 1, sx2]
            v3 = xpower_cumsum[rois_index_sub, :, sy2, sx1 - 1]
            v4 = xpower_cumsum[rois_index_sub, :, sy2, sx2]
            # get real sum by inclusive-exclusive algo
            subroi_power_sum = v4 - v3 - v2 + v1 + 1
            # calculate mean by divide area
            subroi_mean = subroi_power_sum / (ss.unsqueeze(-1))
            # gem = torch.sign(subroi_mean) * torch.pow(torch.abs(subroi_mean), 1.0 / self.p[level])
            gem = torch.pow(torch.maximum(subroi_mean, self.minimumx), 1.0 / self.p[level])
            feat_level = self.proj[level](gem.permute(0, 3, 1, 2))
            roi_feats_list.append(feat_level)

        add_weights = self.weights[target_lvls]
        roi_feats = torch.cat(roi_feats_list, dim=1).reshape(-1, num_levels, 256, output_dim, output_dim)
        roi_feats_weighted = torch.sum(roi_feats * add_weights.reshape(-1, num_levels, 1, 1, 1),dim=1)

        return roi_feats_weighted

