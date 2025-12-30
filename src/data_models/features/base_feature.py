from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import numpy as np
from osgeo.ogr import GeomTransformer
from shapely import LineString, Polygon
from shapely.geometry import shape

from ..processors.feature_processor import FeatureProcessor
from ...utils.geo import check_coords


class BaseFeature(ABC):
    """Base class for all GeoJSON features"""

    _fmt_paths: List[str] = None
    _paths: List[List[Tuple[int,int]]] = None
    _processor: FeatureProcessor = None

    def __init__(self, geojson_feature: Dict, gt: GeomTransformer, picture: np.ndarray, lat_scale=None, lon_scale=None):
        self.feature = geojson_feature
        self.picture = picture
        self.gt = gt
        self.geometry = shape(geojson_feature["geometry"])
        self.properties = geojson_feature.get("properties", {})
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale

    def update_paths(self):
        """ Updates paths """
        self._paths = self.processor.process_feature(self.feature)

    @property
    def paths(self) -> List[List[Tuple[int,int]]]:
        """ Sequence of pixel position """
        if self._paths is None:
            self.update_paths()
        return self._paths

    @paths.setter
    def paths(self, value):
        current =  self._paths
        self._paths = value

    @property
    def fmt_paths(self) -> List[str]:
        """String formated paths"""
        if self._fmt_paths is None:
            self._fmt_paths = self.format_path()
        return self._fmt_paths

    @abstractmethod
    def format_path(self):
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def processor(self) -> FeatureProcessor:
        """Get the processor for this feature type"""
        raise NotImplementedError("Subclasses must implement this method")

    def feature_in_elevation(self, level_range: Tuple[int, int]) -> bool:
        """
        Check if any part of the feature lies within the specified elevation range.

        :param level_range: Tuple of (min_elevation, max_elevation)
        :return: True if any part of the feature is within elevation range, False otherwise
        """

        if isinstance(self.geometry, LineString):
            return check_coords(self.gt, self.picture, list(self.geometry.coords), level_range)

        elif isinstance(self.geometry, Polygon):
            # Check exterior ring
            if check_coords(self.gt, self.picture, list(self.geometry.exterior.coords), level_range):
                return True

            # Check interior rings (holes)
            for interior in self.geometry.interiors:
                if check_coords(self.gt, self.picture, list(interior.coords), level_range):
                    return True
            return False
        elif hasattr(self.geometry, 'geoms'):  # MultiLineString, MultiPolygon, etc.
            # Check all parts
            for part in self.geometry.geoms:
                if self.feature_in_elevation(level_range):
                    return True
            return False
        else:
            return False
