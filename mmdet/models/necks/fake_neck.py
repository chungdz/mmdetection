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
    """

    def __init__(self, 
        in_channels=[256, 512, 1024, 2048],
        out_channels=2048,
        num_outs=1):
        
        super(FakeNeck, self).__init__()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        
        return tuple(inputs[-1])
