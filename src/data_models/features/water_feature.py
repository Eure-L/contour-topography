from typing import List

from data_models.features.base_feature import BaseFeature


class WaterFeature(BaseFeature):
    """Specialized class for water body features"""

    def to_svg_paths(self, pixel_converter) -> List[str]:
        """Convert water geometry to SVG paths"""
        # Implementation would be similar to RoadFeature but with
        # different styling and potentially fill operations
        pass