from typing import List, Tuple

from .base_feature import BaseFeature
from ..processors.line_processor import LineFeatureProcessor


class LineFeature(BaseFeature):
    """Specialized class for road features"""

    @property
    def processor(self) -> LineFeatureProcessor:
        """Get the processor for this feature type"""
        return LineFeatureProcessor(self.gt, self.picture, self.lat_scale)

    def get_layer_keys(self, level_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get elevation layer keys for this feature using the processor"""
        return self.processor.get_layer_key(self.feature, level_ranges)