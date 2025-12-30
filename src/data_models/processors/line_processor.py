from typing import Dict, List, Tuple

from shapely import LineString
from shapely.geometry import shape

from ..path_vector import PathVector
from ...data_models.processors.feature_processor import FeatureProcessor
from ...utils.geo import geo_to_pixel, elevation_at


class LineFeatureProcessor(FeatureProcessor):
    """Processor for generic line features"""

    def process_feature(self, feature: Dict)  -> List[List[Tuple[int, int]]]:
        """Convert line geometry to SVG paths"""

        paths: List[List[Tuple[int, int]]] = []
        geom = feature['geometry']['coordinates']
        geometry = shape(feature['geometry'])

        if geometry.geom_type == "LineString":
            lines = [geom]
        elif geometry.geom_type == "MultiLineString":
            lines = geom
        else:
            return paths

        for line in lines:
            path_parts: List[Tuple[int, int]] = []
            px_prev, py_prev = 0, 0
            path_vector = PathVector(0, 0)

            for lon, lat in line:
                path_vector_prev = path_vector
                px, py = geo_to_pixel(self.gt, lon, lat)
                py = int(py * self.lat_scale)
                path_vector = PathVector(px_prev - px, py_prev - py)
                px_prev, py_prev = px, py

                if path_vector == path_vector_prev:
                    if (px, py) not in path_parts:
                        path_parts[-1] = (px, py)
                    continue

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
