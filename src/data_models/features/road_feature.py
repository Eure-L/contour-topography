from typing import List

from data_models.features.line_feature import LineFeature


class RoadFeature(LineFeature):
    """Specialized class for road features"""


    @property
    def hierarchy(self) -> int:
        """Get road hierarchy level"""
        return int(self.properties.get("HIERARCHY_ID", "0"), 16)

    @property
    def paths(self) -> List[str]:
        """Get road hierarchy level"""
        if self._paths is None:
            self._paths = self.to_svg_paths()
        return self._paths