from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List, Dict

import numpy as np
from osgeo.ogr import GeomTransformer
from shapely.geometry import shape, LineString, Polygon

from utils.geo import geo_to_pixel, elevation_at


class FeatureProcessor(ABC):
    """Abstract base class for all feature processors"""

    def __init__(self, gt: GeomTransformer, picture: np.ndarray, lat_scale: float = 1.0, lon_scale: float = 1.0):
        self.gt = gt
        self.picture = picture
        self.width, self.height = picture.shape
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale

    @abstractmethod
    def process_feature(self, feature: Dict) -> List[str]:
        """Process a single feature and return SVG paths"""
        pass

    @abstractmethod
    def get_layer_key(self, feature: Dict, level_ranges: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Determine which elevation layer this feature belongs to"""
        pass

    def feature_in_elevation(self, feature: Dict, level_range: Tuple[float, float]) -> bool:
        """
        Check if any part of a feature (line or polygon) lies within the specified elevation range.

        Args:
            feature: GeoJSON feature dictionary
            level_range: Tuple of (min_elevation, max_elevation)

        Returns:
            True if any part of the feature is within elevation range, False otherwise
        """
        min_alt, max_alt = level_range
        geom = shape(feature['geometry'])

        def check_coords(coords):
            """Helper function to check if any coordinate is within elevation range"""
            for lon, lat in coords:
                elev = elevation_at(self.gt, self.picture, lon, lat)
                if elev is not None and min_alt <= elev < max_alt:
                    return True
            return False

        if isinstance(geom, LineString):
            return check_coords(list(geom.coords))
        elif isinstance(geom, Polygon):

            # Check exterior ring
            if check_coords(list(geom.exterior.coords)):
                return True

            # Checks interior
            for interior in geom.interiors:
                if check_coords(list(interior.coords)):
                    return True
            return False

        elif hasattr(geom, 'geoms'):  # MultiLineString, MultiPolygon
            for part in geom.geoms:
                if self.feature_in_elevation(part, level_range):
                    return True
            return False
        else:
            return False


class RoadFeatureProcessor(FeatureProcessor):
    """Processor for road features"""

    def process_feature(self, feature: Dict) -> List[str]:
        """Convert road geometry to SVG paths"""
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
                px, py = geo_to_pixel(self.gt,  lon, lat)
                pxpy_str = f"{px},{int(py * self.lat_scale)}"

                # prevents duplicate points in the same road line
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

            for lvl in level_ranges:

                if lvl in feature_ranges:
                    continue

                if elev > lvl[0]:
                    feature_ranges.append(lvl)
                    break

        return feature_ranges


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

        if geom.type == 'MultiPolygon':
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

            for lvl in level_ranges:

                if lvl in feature_ranges:
                    continue

                if elev > lvl[0]:
                    feature_ranges.append(lvl)
                    break

        return feature_ranges
