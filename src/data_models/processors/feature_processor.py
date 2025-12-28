from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import numpy as np
from osgeo.ogr import GeomTransformer
from shapely.geometry import shape, LineString, Polygon

from utils.geo import elevation_at


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
