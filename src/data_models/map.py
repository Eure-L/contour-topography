import json
import logging
import math
import os
from typing import Dict, Tuple, Union, List
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from osgeo import gdal
from osgeo.gdal import Dataset
from osgeo.ogr import GeomTransformer
from shapely import vectorized
from shapely.geometry import shape, Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import transform as shp_transform

from src.utils.colormapping import altitudes_to_rgb_array, altitude_to_rgb
from data_models.color_stop import ColorStop
from utils.colormapping import altitude_to_gray
from defines.color_palettes import ColorPalettes
from defines.road_weights import RoadsWeight

logger = logging.getLogger('map')
logger.setLevel(logging.DEBUG)


class A3:
    width = "297mm"
    height = "420mm"


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


def merge_svgs(svg_files, output_file, width=None, height=None):
    """
    Merge multiple SVG files into a single SVG.

    :param svg_files: list of SVG file paths
    :param output_file: path to save merged SVG
    :param width: width of the merged SVG (optional)
    :param height: height of the merged SVG (optional)
    """
    # Create root SVG element
    merged_svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1"
    )

    if width is not None:
        merged_svg.set("width", str(width))
    if height is not None:
        merged_svg.set("height", str(height))

    for i, svg_file in enumerate(svg_files):
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Wrap content in a group for each SVG
        g = ET.SubElement(merged_svg, "g", id=f"layer{i}")
        for child in root:
            g.append(child)

    # Save merged SVG
    tree = ET.ElementTree(merged_svg)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Merged SVG saved to {output_file}")


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


class Map:
    """
    Interfaces TIF Image file
    """

    _grayscale_picture: np.ndarray = None
    _border_mask: np.ndarray = None
    _color_picture: np.ndarray = None
    _base_layers: Dict = None
    _road_layer: List = []
    _base_road_layers: Dict[Tuple[int, int], List[Tuple[int, str]]] = None
    _width: int = None
    _height: int = None
    _file: str = None
    _borders_geojson: str = None
    _roads_geojson: str = None
    _borders_polygons: List = None
    _roads: List = None
    _ds: Dataset = None
    _gt: GeomTransformer = None
    _geojson_to_tif_transformer = None
    _corners: Dict = None
    _bounding_box: Dict = None
    _name: str = None
    _color_palette: ColorStop = None

    show_contour_strokes = False
    show_roads = True
    road_level = 0x8A
    road_scaling = RoadsWeight.RANKING_1

    def __init__(self, tif_file: str, borders_geojson: str = None, roads_geojson: str = None, name: str = None):
        """

        :param tif_file:            Tif file storing grayscale values
        :param borders_geojson:     optionnal Geojson describing all the borders of the given Map
        :param name:                Optionnal, Sets the name of the map for file saving
        """
        self._file = tif_file
        self._base_layers = {}

        if name is not None:
            self._name = name
        else:
            name = os.path.split(tif_file)[1]
            self._name = ''.join(name.split('.')[:-1])

        if borders_geojson is not None:
            self._borders_geojson = borders_geojson

        if roads_geojson is not None:
            self._roads_geojson = roads_geojson

    def pixel2coord(self, px: int, py: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates (EPSG:4326).

        Args:
            px: Pixel X coordinate
            py: Pixel Y coordinate

        Returns:
            Tuple of (longitude, latitude) coordinates
        """
        gt = self.gt
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        return (x, y)

    def pixel2coord_scaled(self, px: int, py: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to pseudo-metric coordinates with latitude scaling.

        Args:
            px: Pixel X coordinate
            py: Pixel Y coordinate

        Returns:
                    Tuple of (scaled longitude, scaled latitude) coordinates
    """
        gt = self.gt
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        y *= self.lat_scale
        return x, y

    def show_colour_picture(self):
        """
        Display the color elevation map using PIL.
        """
        img = Image.fromarray(self.color_picture, mode='RGB')
        img.show(self.name)

    def elevation_at(self, lon: float, lat: float) -> Union[float, None]:
        """
        Get elevation value at specific geographic coordinates.

        Args:
            lon: Longitude coordinate
            lat: Latitude coordinate

        Returns:
            Elevation value in meters, or None if coordinates are out of bounds
        """
        px, py = self.geo_to_pixel(lon, lat)
        if 0 <= px < self.width and 0 <= py < self.height:
            return float(self.grayscale_picture[py, px])
        return None

    def line_in_elevation(self, line: Union[LineString, BaseGeometry], level_range: Tuple[float, float]) -> bool:
        """
        Check if any part of a line lies within the specified elevation range.

        Args:
            line: Shapely LineString or BaseGeometry object
            level_range: Tuple of (min_elevation, max_elevation)

        Returns:
            True if any part of the line is within elevation range, False otherwise
        """
        min_alt, max_alt = level_range
        coords = list(line.coords)

        for (lon, lat) in coords:
            elev = self.elevation_at(lon, lat)
            inside = (elev is not None and min_alt <= elev < max_alt)
            if inside:
                return True

        return False

    def _save_layer_svg(self, contour: np.ndarray, layer_range: Tuple[Union[int, float], Union[int, float]],
                        save_file: str, for_cut: bool):
        """
        Save a contour layer as an SVG file.

        Args:
            contour: Numpy array of contour points
            layer_range: Elevation range tuple (min, max)
            save_file: Output SVG file path
            for_cut: Whether the SVG is for CNC cutting (affects styling)
        """
        min_alt = self.grayscale_picture.min()
        max_alt = self.grayscale_picture.max()

        if not for_cut:
            r, g, b = altitude_to_rgb(layer_range[0], min_alt, max_alt, self.color_palette)
            svg_color = f"rgb({r},{g},{b})"
            stroke_width_mm = 1
            self.save_map_as_svgs(contour, save_file,
                                  fill_color=svg_color,
                                  stroke_color=svg_color if not self.show_contour_strokes else "black",
                                  fill=True,
                                  stroke_width_mm=stroke_width_mm)
        else:
            gray = 255 - altitude_to_gray(layer_range[0], min_alt, max_alt)
            svg_color = f"rgb({gray},{gray},{gray})"
            stroke_width_mm = 1
            self.save_map_as_svgs(contour, save_file,
                                  fill_color="red",
                                  stroke_color=svg_color if not self.show_contour_strokes else "black",
                                  fill=False,
                                  stroke_width_mm=stroke_width_mm)

    def save_map_as_svgs(self, contours: np.ndarray, filename: str, fill: bool,
                         fill_color: str = "black",
                         stroke_color: str = "black",
                         stroke_width_mm: float = 1):
        """
        Save contours as SVG file with proper scaling and viewbox.

        Args:
            contours: List of contour arrays
            filename: Output SVG file path
            fill: Whether to fill the contours
            fill_color: Stroke/fill color
            stroke_width_mm: Stroke width in millimeters
            stroke_color: Color of the stroke
        """

        stroke_width_mm = round(stroke_width_mm, 1)
        height, width = self.grayscale_picture.shape
        viewbox_height = int(height * self.lat_scale)
        viewbox_width = width

        with open(filename, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{A3.width}" height="{A3.height}" viewBox="0 0 {viewbox_width} {viewbox_height}">\n')
            for contour in contours:
                path_data = "M " + " L ".join(
                    f"{int(x)},{int(y)}" for x, y in contour[:, 0, :]
                )
                path_data = scale_path_y(path_data, self.lat_scale)
                path_data += " Z"
                fill_str = f'fill="{fill_color}"' if fill else f'fill="none"'
                f.write(
                    f'  <path stroke="{stroke_color}" {fill_str} stroke-width="{stroke_width_mm}" d="{path_data}" />\n')
            f.write('</svg>')

    def append_roads_to_svg(self, svg_file: str, road_paths: List[Tuple[int, str]]):
        """
        Append road paths to an existing SVG file.

        Args:
            svg_file: Path to existing SVG file
            road_paths: List of tuples (hierarchy, svg_path_data)
        """

        tree = ET.parse(svg_file)
        root = tree.getroot()

        for road in road_paths:
            hierarchy, d = road
            thickness = self.road_scaling.interpolate(hierarchy)
            thickness = round(thickness, 1)
            path = ET.SubElement(root, "ns0:path", stroke="black", fill="none", **{"stroke-width": f"{thickness}mm"},
                                 d=d)
            path.tail = "\n"

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

    def save_layers(self, save_path: str, combined: bool, for_cut: bool = False, remove_inters=False):
        """
        Save all elevation layers as SVG files.

        Args:
            save_path: Directory to save SVG files
            combined: Whether to combine all layers into one SVG
            for_cut: Whether SVGs are for CNC cutting
            remove_inters: Whether to remove intermediary built layers after combining
        """
        saved_layers = []

        os.makedirs(save_path, exist_ok=True)

        # Save each layer as an individual SVG
        for level_range, contour in self._base_layers.items():
            start, top = level_range
            start, top = int(start), int(top)
            file = os.path.join(save_path, f"{self.name}_{start}-{top}.svg")
            saved_layers.append(file)
            self._save_layer_svg(contour, (start, top), file, for_cut)

            if self.show_roads:
                layer_roads = self._base_road_layers.get(level_range, [])
                if layer_roads:
                    self.append_roads_to_svg(file, layer_roads)

        # Combine layers into a single SVG if requested
        if combined and saved_layers:
            height, width = self.grayscale_picture.shape
            viewbox_height = int(height * self.lat_scale)
            viewbox_width = width
            combined_svg = ET.Element(
                "svg",
                xmlns="http://www.w3.org/2000/svg",
                width=A3.width,
                height=A3.height,
                viewBox=f"0 0 {viewbox_width} {viewbox_height}"
            )

            for layer_file in saved_layers:
                tree = ET.parse(layer_file)
                root = tree.getroot()

                # Append all <g> elements (ignoring the root SVG's viewBox/width/height)
                for g in root.findall(".//{http://www.w3.org/2000/svg}g"):
                    combined_svg.append(g)

                # Append direct <path> elements if roads are stored outside <g>
                for path in root.findall(".//{http://www.w3.org/2000/svg}path"):
                    combined_svg.append(path)

            # Save combined SVG
            output_file = os.path.join(save_path, f"{self.name}.svg")
            ET.ElementTree(combined_svg).write(output_file, encoding="utf-8", xml_declaration=True)
            logger.info(f"Combined SVG saved to {output_file}")

            # Remove intermediary layers if requested
            if remove_inters:
                for layer_file in saved_layers:
                    try:
                        os.remove(layer_file)
                        logger.info(f"Removed intermediary layer: {layer_file}")
                    except OSError as e:
                        logger.error(f"Error removing file {layer_file}: {e}")

    def get_border_mask(self) -> np.ndarray:
        """
        Create a mask where pixels inside borders are 255, outside are 0.

        Returns:
            Numpy array representing the border mask
        """

        height, width = self.grayscale_picture.shape

        if not self.borders_polygons:
            return np.ones((height, width), dtype=np.uint8) * 255

        xs = np.arange(width)
        ys = np.arange(height)
        xx, yy = np.meshgrid(xs, ys)

        gt = self.gt

        # Pixel → lon/lat
        x = gt[0] + xx * gt[1]
        y = gt[3] + yy * gt[5]

        # Apply latitude scale
        y = y * self.lat_scale

        multipoly = MultiPolygon(self.borders_polygons)

        inside = vectorized.contains(multipoly, x, y)

        return (inside.astype(np.uint8) * 255)

    def geo_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Tuple of (pixel_x, pixel_y) coordinates
        """

        gt = self.gt

        px = (lon - gt[0]) / gt[1]
        py = (lat - gt[3]) / gt[5]

        return int(px), int(py)

    def line_to_svg_path(self, line: Union[LineString, BaseGeometry]) -> str:
        """
        Convert a Shapely line to SVG path data string.

        Args:
            line: Shapely LineString or BaseGeometry

        Returns:
            SVG path data string
        """

        parts = []
        for lon, lat in line.coords:
            px, py = self.geo_to_pixel(lon, lat)
            parts.append(f"{px},{py}")
        d = f"M {' L '.join(parts)}"
        d = scale_path_y(d, self.lat_scale)
        return d

    def road_to_svg_paths(self, feature: Dict) -> List[str]:
        """
        Convert GeoJSON road feature to SVG path strings.

        Args:
            feature: GeoJSON feature dictionary

        Returns:
            List of SVG path data strings
        """

        geom = shape(feature["geometry"])
        paths = []

        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            return paths

        for line in lines:
            path_parts = []
            for lon, lat in line.coords:
                px, py = self.geo_to_pixel(lon, lat)

                if px < -10 or py < -10 or px > self.width + 10 or py > self.height + 10:
                    continue

                path_parts.append(f"{px},{py}")

            if len(path_parts) > 1:
                d = "M " + " L ".join(path_parts)
                paths.append(d)

        return paths

    def compute_road_layers(self):
        """
        Compute road layers for each elevation layer.

        Populates self._base_road_layers with road segments that fall within
        each elevation layer's range.
        """

        self._base_road_layers = {lr: [] for lr in self._base_layers.keys()}
        level_ranges = list(self._base_layers.keys())
        next_level_ranges = level_ranges[1:]
        next_level_ranges.append(level_ranges[-1])

        for idx, level_range in enumerate(level_ranges):
            start, end = level_range
            nex_start, nex_end = next_level_ranges[idx]

            for feature in self.roads:

                hierarchy = int(feature['properties']['HIERARCHY_ID'], 16)
                if hierarchy > self.road_level:
                    continue

                geom = shape(feature["geometry"])
                if geom.geom_type == "LineString":
                    lines = [geom]
                elif geom.geom_type == "MultiLineString":
                    lines = list(geom.geoms)
                else:
                    continue

                for line in lines:
                    if self.line_in_elevation(line, (start, nex_start)):
                        svg_path = self.line_to_svg_path(line)
                        self._base_road_layers[level_range].append((hierarchy, svg_path))

    def compute_base_layer(self, level_range: Tuple[Union[float, int], Union[float, int]]):
        """
        Compute contour for a specific elevation range.

        Args:
            level_range: Tuple of (min_elevation, max_elevation)
        """
        mask = np.zeros_like(self.grayscale_picture, dtype=np.uint8)

        if len(self.borders_polygons) == 0:
            mask[(self.grayscale_picture >= level_range[0]) &
                 (self.grayscale_picture < level_range[1])] = 255
        else:
            mask[(self.grayscale_picture >= level_range[0]) &
                 (self.grayscale_picture < level_range[1])] = 255
            mask[self.border_mask != 255] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out contours with fewer than 20 points
        contours = [cnt for cnt in contours if len(cnt) >= 20]

        self._base_layers[level_range] = contours

    def compute_all_layers(self, level_steps: List[int]):
        """
        Compute all elevation layers based on given level steps.

        Args:
            level_steps: List of elevation values defining layer boundaries
        """

        self._base_layers = {}

        for idx, _ in enumerate(level_steps):
            if idx == len(level_steps) - 1:
                break

            print(f"Processing Level {level_steps[idx]}m")
            level_range = (level_steps[idx - 1], level_steps[-1])
            self.compute_base_layer(level_range)

        if self.show_roads:
            self.compute_road_layers()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def border_mask(self):
        if self._border_mask is None:
            self._border_mask = self.get_border_mask()
        return self._border_mask

    @property
    def borders_polygons(self):
        if self._borders_polygons is None:
            self._borders_polygons = []

            if self._borders_geojson is None:
                return self._borders_polygons

            with open(self._borders_geojson, 'r') as f:
                geojson = json.load(f)

            scale = self.lat_scale

            for feature in geojson['features']:
                geom = shape(feature['geometry'])
                # Scale latitude ONLY
                geom = shp_transform(
                    lambda lon, lat: (lon, lat * scale),
                    geom
                )
                self._borders_polygons.append(geom)

        return self._borders_polygons

    @property
    def roads(self):
        if self._roads is None:
            self._roads = []
            if self._roads_geojson is None:
                return self._roads

            with open(self._roads_geojson, 'r') as f:
                geojson = json.load(f)
            self._roads = geojson['features']

        return self._roads

    @property
    def width(self):
        if self._width is None:
            _, width = self.grayscale_picture.shape
            self._width = width
        return self._width

    @property
    def height(self):
        if self._height is None:
            _, height = self.grayscale_picture.shape
            self._height = height
        return self._height

    @property
    def grayscale_picture(self) -> np.ndarray:
        if self._grayscale_picture is None:
            self._grayscale_picture = cv2.imread(self._file, cv2.IMREAD_UNCHANGED)
        return self._grayscale_picture

    @property
    def color_picture(self):
        if self._color_picture is None:
            self._color_picture = altitudes_to_rgb_array(self.grayscale_picture)
        return self._color_picture

    @property
    def file(self) -> str:
        return self._file

    @property
    def ds(self) -> Dataset:
        if self._ds is None:
            self._ds = gdal.Open(self.file, gdal.GA_ReadOnly)

        return self._ds

    @property
    def gt(self):
        if self._gt is None:
            self._gt = self.ds.GetGeoTransform()
        return self._gt

    @property
    def corners(self) -> Dict[str, Tuple[float]]:
        if self._corners is None:
            self._corners = self._get_corners()
        return self._corners

    @property
    def color_palette(self) -> ColorStop:
        if self._color_palette is None:
            self._color_palette = ColorPalettes.BROWN_1
        return self._color_palette

    @color_palette.setter
    def color_palette(self, value: ColorStop):
        self._color_palette = value

    @property
    def bounding_box(self) -> Dict[str, float]:
        if self._bounding_box is None:
            self._bounding_box = {
                "north_latitude": self.corners["upper_left"][1],
                "south_latitude": self.corners["lower_left"][1],
                "west_longitude": self.corners["upper_left"][0],
                "east_longitude": self.corners["upper_right"][0],
            }

        return self._bounding_box

    @property
    def north_latitude(self):
        return self.bounding_box["north_latitude"]

    @property
    def south_latitude(self):
        return self.bounding_box["south_latitude"]

    @property
    def east_longitude(self):
        return self.bounding_box["east_longitude"]

    @property
    def west_longitude(self):
        return self.bounding_box["west_longitude"]

    def _get_corners(self):
        """Return the geographic coordinates of the 4 corners and center."""
        xsize = self.ds.RasterXSize
        ysize = self.ds.RasterYSize

        # Pixel to geo coordinates
        corners = {
            "upper_left": self.pixel2coord(0, 0),
            "upper_right": self.pixel2coord(xsize, 0),
            "lower_left": self.pixel2coord(0, ysize),
            "lower_right": self.pixel2coord(xsize, ysize),
            "center": self.pixel2coord(xsize // 2, ysize // 2)
        }

        return corners

    @property
    def lat_scale(self):
        """
        Scale factor to compensate for EPSG:4326 latitude distortion.
        """
        lat0 = self.corners["center"][1]  # degrees
        scale = 1.0 / math.cos(math.radians(lat0))
        return scale

    def debug_scaling(self):
        px1, py1 = self.geo_to_pixel(
            self.corners["center"][0],
            self.corners["center"][1]
        )

        px2, py2 = self.geo_to_pixel(
            self.corners["center"][0],
            self.corners["center"][1] + 0.01
        )

        print("Δpy:", py2 - py1)
