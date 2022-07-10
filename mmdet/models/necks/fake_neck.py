# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


@NECKS.register_module()
class FakeNeck(BaseModule):
    r"""
    input is tuple of features
    output is the last one features in tuple
    """

    def __init__(self, 
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=dict(
            type='Xavier', layer='Conv2d', distribution='uniform')):

        super(FakeNeck, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        llen = len(laterals)
        if self.num_outs > llen:
            for _ in range(self.num_outs - llen):
                laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))
        
        return tuple(laterals)

