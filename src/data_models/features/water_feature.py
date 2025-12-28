from typing import List, Tuple

from .base_feature import BaseFeature
from ..processors.water_processor import WaterFeatureProcessor


class WaterFeature(BaseFeature):
    """Specialized class for water body features"""


    @property
    def processor(self) -> WaterFeatureProcessor:
        """Get the processor for this feature type"""
        return WaterFeatureProcessor(self.gt, self.picture, self.lat_scale)


    def get_layer_keys(self, level_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get elevation layer keys for this feature using the processor"""
        return self.processor.get_layer_key(self.feature, level_ranges)
