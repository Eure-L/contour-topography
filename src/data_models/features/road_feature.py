from typing import List

from shapely import LineString

from data_models.features.base_feature import BaseFeature
from utils.geo import pixel2coord_scaled, geo_to_pixel


class RoadFeature(BaseFeature):
    """Specialized class for road features"""
    _paths: List[str] = None
    lat_scaler = None
    lon_scaler = None

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

                path_parts.append(f"{px},{int(py * self.lat_scale)}")

            if len(path_parts) > 1:
                d = "M " + " L ".join(path_parts)
                paths.append(d)

        return paths

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