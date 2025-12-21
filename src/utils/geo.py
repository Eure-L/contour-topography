from typing import Tuple, List, Union

from osgeo.ogr import GeomTransformer
from shapely import Point, LineString
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry


def pixel2coord(gt: GeomTransformer, px: int, py: int) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates (EPSG:4326).

    Args:
        px: Pixel X coordinate
        py: Pixel Y coordinate

    Returns:
        Tuple of (longitude, latitude) coordinates
    """
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return (x, y)


def pixel2coord_scaled(gt: GeomTransformer, px: int, py: int, lat_scale: float) -> Tuple[float, float]:
    """
    Convert pixel coordinates to pseudo-metric coordinates with latitude scaling.

    Args:
        px: Pixel X coordinate
        py: Pixel Y coordinate

    Returns:
        Tuple of (scaled longitude, scaled latitude) coordinates

    """
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    y *= lat_scale

    return x, y


def geo_to_pixel(gt: GeomTransformer, lon: float, lat: float) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates.

    Args:
        lon: Longitude
        lat: Latitude

    Returns:
        Tuple of (pixel_x, pixel_y) coordinates
    """

    px = (lon - gt[0]) / gt[1]
    py = (lat - gt[3]) / gt[5]

    return int(px), int(py)


def point_in_border(point: Point, borders: List[shape]) -> bool:
    """
    Check if a point is inside any of the given border polygons.

    Args:
        point: Shapely Point object to check
        borders: List of Shapely polygon objects representing borders

    Returns:
        True if point is inside any border, False otherwise
    """
    for border in borders:
        if border.contains(point):
            return True
    return False


def scale_path_y(path: str, lat_scale: float) -> str:
    """
    Scale the Y coordinates in an SVG path string by the given latitude scale factor.

    Args:
        path: SVG path string in format "M x1,y1 L x2,y2 ..."
        lat_scale: Scale factor to apply to Y coordinates

    Returns:
        SVG path string with scaled Y coordinates
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

    Args:
        line: Shapely LineString or BaseGeometry

    Returns:
        SVG path data string
    """

    parts = []
    for lon, lat in line.coords:
        px, py = geo_to_pixel(gt, lon, lat)
        parts.append(f"{px},{py}")
    d = f"M {' L '.join(parts)}"

    return d
