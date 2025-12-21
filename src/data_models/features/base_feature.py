from typing import List, Dict

from osgeo.ogr import GeomTransformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


class BaseFeature:
    """Base class for all GeoJSON features"""

    def __init__(self, geojson_feature: Dict, gt:GeomTransformer, lat_scale=None, lon_scale=None):
        self.feature = geojson_feature
        self.gt = gt
        self.geometry = shape(geojson_feature["geometry"])
        self.properties = geojson_feature.get("properties", {})
        self.lat_scale = lat_scale
        self.lon_scale = lon_scale

    def to_svg_paths(self) -> List[str]:
        """Convert geometry to SVG path strings"""
        raise NotImplementedError("Subclasses must implement this method")
