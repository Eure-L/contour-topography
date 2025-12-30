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
        """Get road id"""
        return self.properties.get("ROAD_ID", 0)

    @property
    def id(self) -> int:
        """Gets unique object id"""
        return self.properties.get("ID", 0)

    @property
    def processor(self) -> RoadFeatureProcessor:
        """Get the processor for this feature type"""
        return RoadFeatureProcessor(self.gt, self.picture, self.lat_scale)

    def get_layer_keys(self, level_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Get elevation layer keys for this feature using the processor"""
        return self.processor.get_layer_key(self.feature, level_ranges)

    def format_path(self):
        """ Formats Feature's path for SVG data field """
        paths_str = []
        for path in self.paths:
            pxpy_str_list = list(map(lambda pxpy: f"{pxpy[0]},{pxpy[1]}", path))
            path_str = "M " + " L ".join(pxpy_str_list)
            paths_str.append(path_str)
        return paths_str