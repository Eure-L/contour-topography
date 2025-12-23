from typing import List, Dict

from osgeo.ogr import GeomTransformer

from data_models.features.base_feature import BaseFeature
from utils.geo import geo_to_pixel


class WaterFeature(BaseFeature):
    """Specialized class for water body features"""

    def to_svg_paths(self) -> List[str]:
        """Convert water geometry to SVG paths"""
        paths = []
        geom = self.geometry

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            return paths

        for polygon in polygons:
            # Process exterior ring
            path_parts = []
            for lon, lat in polygon.exterior.coords:
                px, py = geo_to_pixel(self.gt, lon, lat)
                pxpy_str = f"{px},{int(py * self.lat_scale)}"
                if pxpy_str not in path_parts:
                    path_parts.append(pxpy_str)

            if len(path_parts) > 1:
                d = "M " + " L ".join(path_parts) + " Z"
                paths.append(d)

            # Process interior rings (holes)
            for interior in polygon.interiors:
                path_parts = []
                for lon, lat in interior.coords:
                    px, py = geo_to_pixel(self.gt, lon, lat)
                    pxpy_str = f"{px},{int(py * self.lat_scale)}"
                    if pxpy_str not in path_parts:
                        path_parts.append(pxpy_str)

                if len(path_parts) > 1:
                    d = "M " + " L ".join(path_parts) + " Z"
                    paths.append(d)

        return paths
