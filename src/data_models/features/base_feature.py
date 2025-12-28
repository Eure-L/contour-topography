from typing import List, Dict

import numpy as np
from osgeo.ogr import GeomTransformer
from shapely.geometry import shape

from ..processors.feature_processor import FeatureProcessor


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

