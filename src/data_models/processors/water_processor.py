from typing import Dict, List, Tuple

from shapely.geometry import shape

from ...data_models.processors.feature_processor import FeatureProcessor
from ...utils.geo import geo_to_pixel, elevation_at


class WaterFeatureProcessor(FeatureProcessor):
    """Processor for water body features"""

    def process_feature(self, feature: Dict) -> List[List[Tuple[int, int]]]:
        """Convert water geometry to SVG paths"""
        paths: List[List[Tuple[int, int]]] = []
        geom = shape(feature['geometry'])

        if geom.geom_type == "Polygon":
            polygons = [geom]
        elif geom.geom_type == "MultiPolygon":
            polygons = list(geom.geoms)
        else:
            return paths

        for polygon in polygons:
            # Process exterior ring
            path_parts: List[Tuple[int, int]] = []
            for lon, lat in polygon.exterior.coords:
                px, py = geo_to_pixel(self.gt, lon, lat)
                py = int(py * self.lat_scale)
                if (px, py) not in path_parts:
                    path_parts.append((px, py))

            if len(path_parts) > 1:
                paths.append(path_parts)

            # Process interior rings (holes)
            for interior in polygon.interiors:

                path_parts: List[Tuple[int, int]] = []
                for lon, lat in interior.coords:
                    px, py = geo_to_pixel(self.gt, lon, lat)
                    py = int(py * self.lat_scale)
                    if (px, py) not in path_parts:
                        path_parts.append((px, py))

                if len(path_parts) > 1:
                    paths.append(path_parts)

        return paths

    def get_layer_key(self, feature: Dict, level_ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """ Determine all possible elevation layers based on road elevation at each point """

        geom = shape(feature['geometry'])
        feature_ranges = []

        if geom.geom_type == 'MultiPolygon':
            polygons = feature['geometry']['coordinates'][0]
        else:
            polygons = [feature['geometry']['coordinates'][0]]

        for polygon in polygons:
            for point in polygon:
                elev = elevation_at(self.gt, self.picture, point[0], point[1])
                if elev is None:
                    break

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
