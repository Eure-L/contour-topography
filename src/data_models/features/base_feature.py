from typing import List, Dict, Tuple

import numpy as np
from osgeo.ogr import GeomTransformer
from shapely import LineString, Polygon
from shapely.geometry import shape

from ..processors.feature_processor import FeatureProcessor
from ...utils.geo import check_coords


class BaseFeature:
    """Base class for all GeoJSON features"""

    _paths : List[str] = None
    _processor : FeatureProcessor = None

    def __init__(self, geojson_feature: Dict, gt:GeomTransformer, picture: np.ndarray, lat_scale=None, lon_scale=None):
        self.feature = geojson_feature
        self.picture = picture
        self.gt = gt
        self.geometry = shape(geojson_feature["geometry"])
        self.properties = geojson_feature.get("properties", {})
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale

    @property
    def paths(self) -> List[str]:
        """Get road hierarchy level"""
        if self._paths is None:
            self._paths = self.processor.process_feature(self.feature)
        return self._paths

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
