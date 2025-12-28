from typing import Dict, List, Tuple

from shapely.geometry import shape

from data_models.processors.feature_processor import FeatureProcessor
from utils.geo import geo_to_pixel, elevation_at


class WaterFeatureProcessor(FeatureProcessor):
    """Processor for water body features"""

    def process_feature(self, feature: Dict) -> List[str]:
        """Convert water geometry to SVG paths"""
        paths = []
        geom = shape(feature['geometry'])

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
                    continue
                for lvl in level_ranges:
                    if lvl in feature_ranges:
                        continue
                    if elev > lvl[0]:
                        feature_ranges.append(lvl)
                        break

        return feature_ranges
