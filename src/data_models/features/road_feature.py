from typing import List, Tuple

from ..processors.road_processor import RoadFeatureProcessor
from .line_feature import LineFeature


class RoadFeature(LineFeature):
    """Specialized class for road features"""


    @property
    def hierarchy(self) -> int:
        """Get road hierarchy level"""
        return int(self.properties.get("HIERARCHY_ID", "0"), 16)

    @property
    def road_id(self) -> int:
        """Get road hierarchy level"""
        return self.properties.get("ID", 0)

    @property
    def processor(self) -> RoadFeatureProcessor:
        """Get the processor for this feature type"""
        return RoadFeatureProcessor(self.gt, self.picture, self.lat_scale)

    def get_layer_keys(self, level_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get elevation layer keys for this feature using the processor"""
        return self.processor.get_layer_key(self.feature, level_ranges)