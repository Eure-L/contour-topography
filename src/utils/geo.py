from typing import TYPE_CHECKING, Tuple, List, Union

import numpy as np

from osgeo.ogr import GeomTransformer
from shapely import Point, LineString
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

def pixel2coord(gt: GeomTransformer, px: int, py: int) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates (EPSG:4326).

    :param gt: GeomTransformer object
    :param px: Pixel X coordinate
    :param py: Pixel Y coordinate
    :return: Tuple of (longitude, latitude) coordinates
    """
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return (x, y)

def pixel2coord_scaled(gt: GeomTransformer, px: int, py: int, lat_scale: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to pseudo-metric coordinates with latitude scaling.

    :param gt: GeomTransformer object
    :param px: Pixel X coordinate
    :param py: Pixel Y coordinate
    :param lat_scale: Scale factor to apply to Y coordinates
    :return: Tuple of (scaled longitude, scaled latitude) coordinates
    """
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    y *= lat_scale

    return x, y

def geo_to_pixel(gt: GeomTransformer, lon: float, lat: float) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates.

    :param gt: GeomTransformer object
    :param lon: Longitude
    :param lat: Latitude
    :return: Tuple of (pixel_x, pixel_y) coordinates
    """

    px = (lon - gt[0]) / gt[1]
    py = (lat - gt[3]) / gt[5]

    return int(px), int(py)

def point_in_border(point: Point, borders: List[shape]) -> bool:
    """
    Check if a point is inside any of the given border polygons.

    :param point: Shapely Point object to check
    :param borders: List of Shapely polygon objects representing borders
    :return: True if point is inside any border, False otherwise
    """
    for border in borders:
        if border.contains(point):
            return True
    return False

def scale_path_y(path: str, lat_scale: float) -> str:
    """
    Scale the Y coordinates in an SVG path string by the given latitude scale factor.

    :param path: SVG path string in format "M x1,y1 L x2,y2 ..."
    :param lat_scale: Scale factor to apply to Y coordinates
    :return: SVG path string with scaled Y coordinates
    """
    parts = path.split()
    new_parts = []
    for part in parts:
        if ',' in part:
            x, y = part.split(',')
            new_parts.append(f"{x},{int(float(y) * lat_scale)}")
        else:
            new_parts.append(part)
    return " ".join(new_parts)

def line_to_svg_path(gt: GeomTransformer, line: Union[LineString, BaseGeometry]) -> str:
    """
    Convert a Shapely line to SVG path data string.

    :param gt: GeomTransformer object
    :param line: Shapely LineString or BaseGeometry
    :return: SVG path data string
    """

    parts = []
    for lon, lat in line.coords:
        px, py = geo_to_pixel(gt, lon, lat)
        parts.append(f"{px},{py}")
    d = f"M {' L '.join(parts)}"

    return d

def elevation_at(gt: GeomTransformer, picture: np.ndarray, lon: float, lat: float) -> Union[float, None]:
    """
    Get elevation value at specific geographic coordinates.

    :param gt: GeomTransformer object
    :param picture: 2d array object to check elevation on
    :param lon: Longitude coordinate
    :param lat: Latitude coordinate
    :return: Elevation value in meters, or None if coordinates are out of bounds
    """
    px, py = geo_to_pixel(gt, lon, lat)
    width, height = picture.shape
    if 0 <= px < width and 0 <= py < height:
        return float(picture[py, px])

    return None

def check_coords(gt: GeomTransformer, picture: np.ndarray, coords: List[Tuple[float, float]],
                 level_range: Tuple[int, int]):
    """Helper function to check if any coordinate is within elevation range"""
    min_alt, max_alt = level_range
    for lon, lat in coords:
        elev = elevation_at(gt, picture, lon, lat)
        if elev is not None and min_alt <= elev < max_alt:
            return True
    return False
