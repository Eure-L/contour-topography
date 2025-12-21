from typing import Tuple

from osgeo.ogr import GeomTransformer


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
