from typing import Dict, List, Tuple

from shapely import LineString
from shapely.geometry import shape

from data_models.processors.feature_processor import FeatureProcessor
from utils.geo import geo_to_pixel, elevation_at


class LineFeatureProcessor(FeatureProcessor):
    """Processor for generic line features"""

    def process_feature(self, feature: Dict) -> List[str]:
        """Convert line geometry to SVG paths"""
        paths = []
        geom = feature['geometry']['coordinates']
        geometry = shape(feature['geometry'])

        if geometry.geom_type == "LineString":
            lines = [geom]
        elif geometry.geom_type == "MultiLineString":
            lines = geom
        else:
            return paths

        for line in lines:
            path_parts = []
            for lon, lat in line:
                px, py = geo_to_pixel(self.gt, lon, lat)
                pxpy_str = f"{px},{int(py * self.lat_scale)}"

                # prevents duplicate points in the sameline
                if pxpy_str not in path_parts:
                    path_parts.append(pxpy_str)

            if len(path_parts) > 1:
                d = "M " + " L ".join(path_parts)
                paths.append(d)

        return paths

    def get_layer_key(self, feature: Dict, level_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """ Determine all possible elevation layers based on road elevation at each point """

        geom = shape(feature['geometry'])
        feature_ranges = []

        road_line = geom if isinstance(geom, LineString) else geom.geoms[0]
        for point in road_line.coords:
            elev = elevation_at(self.gt, self.picture, point[0], point[1])
            if elev is None:
                continue

            for idx, lvl in enumerate(level_ranges):
                next_lvl = level_ranges[idx + 1] if idx < len(level_ranges) - 1 else lvl
                if lvl in feature_ranges:
                    continue

                if next_lvl[0] >= elev >= lvl[0]:
                    feature_ranges.append(lvl)
                    break

                if next_lvl[0] >= lvl[0] > elev:
                    break

        return feature_ranges
