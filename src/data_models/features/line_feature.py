from typing import List

from shapely import LineString

from data_models.features.base_feature import BaseFeature
from utils.geo import pixel2coord_scaled, geo_to_pixel


class LineFeature(BaseFeature):
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
                pxpy_str = f"{px},{int(py * self.lat_scale)}"

                # prevents duplicate points in the same road line
                if pxpy_str not in path_parts:
                    path_parts.append(pxpy_str)

            if len(path_parts) > 1:
                d = "M " + " L ".join(path_parts)
                paths.append(d)

        return paths


    @property
    def paths(self) -> List[str]:
        """Get road hierarchy level"""
        if self._paths is None:
            self._paths = self.to_svg_paths()
        return self._paths