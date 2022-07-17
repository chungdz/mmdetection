# Copyright (c) OpenMMLab. All rights reserved.
from .base_roi_extractor import BaseRoIExtractor
from .generic_roi_extractor import GenericRoIExtractor
from .single_level_roi_extractor import SingleRoIExtractor
from .single_level_roi_extractor_gem import SingleRoIExtractorGeM

__all__ = ['BaseRoIExtractor', 'SingleRoIExtractor', 'GenericRoIExtractor', 'SingleRoIExtractorGeM']
