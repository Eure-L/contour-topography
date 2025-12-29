import logging
from typing import Dict, List, Tuple, Set

from shapely import LineString
from shapely.geometry import shape

from ...data_models.processors.feature_processor import FeatureProcessor
from ...utils.geo import geo_to_pixel, elevation_at

logger = logging.getLogger()


class RoadFeatureProcessor(FeatureProcessor):
    """Processor for road features"""

    def process_feature(self, feature: Dict) -> List[str]:
        """Convert road geometry to SVG paths"""

        paths: List[List[Tuple[int, int]]] = []
        geom = feature['geometry']['coordinates']
        geometry = shape(feature['geometry'])

        if geometry.geom_type == "LineString":
            lines = [geom]
        elif geometry.geom_type == "MultiLineString":
            lines = geom
        else:
            return []

        for line in lines:
            path_parts: List[Tuple[int, int]] = []
            for lon, lat in line:
                px, py = geo_to_pixel(self.gt, lon, lat)
                py = int(py * self.lat_scale)

                # prevents duplicate points in the same road line
                if (px, py) not in path_parts:
                    path_parts.append((px, py))

            if len(path_parts) <= 1:
                continue

            is_extent = False

            for idx, existing_path in enumerate(paths):

                if existing_path[0] == path_parts[-1]:
                    existing_path = path_parts[:-1] + existing_path
                    is_extent = True

                elif existing_path[-1] == path_parts[0]:
                    existing_path = existing_path + path_parts[1:]
                    is_extent = True

                elif existing_path[-1] == path_parts[-1]:
                    truncated_path = existing_path[:-1]
                    truncated_path.reverse()
                    existing_path = path_parts + truncated_path
                    is_extent = True

                elif existing_path[0] == path_parts[0]:
                    truncated_path = existing_path[1:]
                    truncated_path.reverse()
                    existing_path = truncated_path + path_parts
                    is_extent = True
                else:
                    continue

                paths[idx] = existing_path
                break

            if not is_extent:
                paths.append(path_parts)

        paths_str = []
        for path in paths:
            pxpy_str_list = list(map(lambda pxpy: f"{pxpy[0]},{pxpy[1]}", path))
            path_str = "M " + " L ".join(pxpy_str_list)
            paths_str.append(path_str)

        # d_list = list(map(lambda path_str: "M " + " L ".join(path_str), path_str_list))
        return paths_str

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
