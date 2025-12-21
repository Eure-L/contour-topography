from typing import List

from data_models.features.base_feature import BaseFeature
from utils.geo import geo_to_pixel


class WaterFeature(BaseFeature):
    """Specialized class for water body features"""

    def to_svg_paths(self) -> List[str]:
        """Convert road geometry to SVG paths"""
        paths = []
        geom = self.feature['geometry']['coordinates']
        if self.geometry.geom_type == "LineString":
            lines = [geom]
        elif self.geometry.geom_type == "MultiLineString":
            lines = self.feature['geometry']['coordinates']
        else:
            return paths

        for line in lines:
            path_parts = []
            for lon, lat in line:
                px, py = geo_to_pixel(self.gt, lon, lat)
                pxpy_str = f"{px},{int(py * self.lat_scale)}"

                # prevents duplicate points in the same road line
                if pxpy_str not in path_parts:
                    path_parts.append(pxpy_str)

            if len(path_parts) > 1:
                d = "M " + " L ".join(path_parts)
                paths.append(d)

        return paths