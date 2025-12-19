import json
import os
from xml.etree import ElementTree as ET
from typing import Dict, Tuple, Union, List
from PIL import Image
import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
import svgutils.transform as st
from osgeo.ogr import GeomTransformer
from shapely import vectorized
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon

from src.utils.colormapping import altitudes_to_rgb_array, altitude_to_rgb
from shapely.geometry import shape, Point
import logging

from utils.colormapping import altitude_to_gray
from utils.colors import BROWN_1, BROWN_2
from utils.roads_weights import RoadsWeight

logger = logging.getLogger('map')
logger.setLevel(logging.DEBUG)


def pixel2coord(gt, px, py):
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return (x, y)


def save_map_as_svgs(contours, width, height, filename, fill: bool, color="black", stroke_width_mm: float = 1):
    """Save contours as SVG."""
    with open(filename, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        for contour in contours:
            path_data = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in contour[:, 0, :])
            path_data += " Z"
            fill_str = f'fill="{color}"' if fill else f'fill="none"'
            f.write(f'  <path stroke="{color}" {fill_str} stroke-width="{stroke_width_mm}" d="{path_data}" />\n')
        f.write('</svg>')


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


def point_in_border(point: Point, borders: List[shape]):
    """
    Checks wether a point is inside a list of borders.
    :param point:
    :param borders:
    :return:
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
    _corners: Dict = None
    _bounding_box: Dict = None
    _name: str = None

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

    def show_colour_picture(self):
        img = Image.fromarray(self.color_picture, mode='RGB')
        img.show(self.name)

    def elevation_at(self, lon, lat):
        px, py = self.geo_to_pixel(lon, lat)
        if 0 <= px < self.width and 0 <= py < self.height:
            return float(self.grayscale_picture[py, px])
        return None

    def line_in_elevation(self, line: Union[LineString, BaseGeometry], level_range):
        """
        Returns a list of LineString segments from 'line' whose elevation lies inside level_range.
        """

        min_alt, max_alt = level_range
        coords = list(line.coords)

        for (lon, lat) in coords:
            elev = self.elevation_at(lon, lat)
            inside = (elev is not None and min_alt <= elev < max_alt)
            if inside:
                return True

        return False

    def _save_layer_svg(self, contour, layer_range: Tuple[Union[int, float], Union[int, float]], save_file: str,
                        for_cut: bool):
        """
        Saves a given layer altitude range as SVG

        :param save_file: dst save file
        :param for_cut:
        :return:
        """
        height, width = self.grayscale_picture.shape
        min_alt = self.grayscale_picture.min()
        max_alt = self.grayscale_picture.max()

        stops = BROWN_1

        if not for_cut:
            r, g, b = altitude_to_rgb(layer_range[0], min_alt, max_alt, stops=stops.stops)
            svg_color = f"rgb({r},{g},{b})"
            stroke_width_mm = 1
            save_map_as_svgs(contour, width, height, save_file, color=svg_color, fill=True,
                             stroke_width_mm=stroke_width_mm)
        else:
            gray = 255 - altitude_to_gray(layer_range[0], min_alt, max_alt)
            svg_color = f"rgb({gray},{gray},{gray})"
            stroke_width_mm = 1
            save_map_as_svgs(contour, width, height, save_file, color="red", fill=False,
                             stroke_width_mm=stroke_width_mm)

    def save_roads_svg(self, filename):
        width, height = self.width, self.height

        with open(filename, "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')

            for d in self._road_layer:
                f.write(f'  <path stroke="black" fill="none" stroke-width="1mm" d="{d}"/>\n')
            f.write('\n</svg>')

    def append_roads_to_svg(self, svg_file, road_paths: List[Tuple[int, str]]):
        """
        Append road SVG <path> elements to an existing SVG file.
        """

        tree = ET.parse(svg_file)
        root = tree.getroot()

        for road in road_paths:
            hierarchy, d = road
            thickness = self.road_scaling.interpolate(hierarchy)
            path = ET.SubElement(root, "ns0:path", stroke="black", fill="none", **{"stroke-width": f"{thickness}mm"}, d=d)
            path.tail = "\n"

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

    def save_layers(self, save_path: str, combined: bool, for_cut: bool = False):
        """
        Saves all computed layers to SVGs or a single SVG.

        :param for_cut:     Wether its for CNC cutting
        :param save_path:   Dst path
        :param combined:    Wether to combine them all SVGs
        :return:
        """

        saved_layers = []

        for level_range, contour in self._base_layers.items():
            start, top = level_range
            start, top = int(start), int(top)
            file = os.path.join(save_path, f"{self.name}_{start}-{top}.svg")
            saved_layers.append(file)

            self._save_layer_svg(contour, (start, top), file, for_cut)
            layer_roads = self._base_road_layers[level_range]
            if layer_roads:
                self.append_roads_to_svg(file, layer_roads)

        if self.show_roads:
            road_svg = os.path.join(save_path, f"{self.name}_roads.svg")
            self.save_roads_svg(road_svg)
            saved_layers.append(road_svg)

        if combined:
            first_layer = saved_layers[0]
            first_svg = st.fromfile(first_layer)
            for next_layer in saved_layers[1:]:
                next_svg = st.fromfile(next_layer)
                first_svg.append(next_svg)

            output_file = os.path.join(save_path, self.name + '.svg')
            logger.log(logging.INFO, f"Combined layer => {output_file}")
            first_svg.save(output_file)

    def get_border_mask(self):
        """
        Create a mask where pixels inside any of the given borders are 1, otherwise 0.
        Vectorized implementation using shapely.vectorized.contains.
        """

        height, width = self.grayscale_picture.shape
        gt = self.gt

        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        lon = gt[0] + xx * gt[1] + yy * gt[2]
        lat = gt[3] + xx * gt[4] + yy * gt[5]

        # Combine all borders into one multipolygon
        if not self.borders_polygons:
            return np.ones((height, width), dtype=np.uint8)

        multipoly = MultiPolygon(self.borders_polygons)

        # Vectorized containment check
        mask = vectorized.contains(multipoly, lon, lat)

        return mask.astype(np.uint8) * 255

    def geo_to_pixel(self, lon, lat):
        """
        Convert geographic coordinates (lon, lat) to pixel coordinates (px, py)
        using the map geotransform.
        """
        gt = self.gt
        inv_det = 1 / (gt[1] * gt[5] - gt[2] * gt[4])

        px = int((gt[5] * (lon - gt[0]) - gt[2] * (lat - gt[3])) * inv_det)
        py = int((-gt[4] * (lon - gt[0]) + gt[1] * (lat - gt[3])) * inv_det)

        return px, py

    def line_to_svg_path(self, line: Union[LineString, BaseGeometry]):
        parts = []
        for lon, lat in line.coords:
            px, py = self.geo_to_pixel(lon, lat)
            parts.append(f"{px},{py}")
        return f"M {' L '.join(parts)}"

    def road_to_svg_paths(self, feature):
        """
        Convert a GeoJSON road feature to one or more SVG <path> strings.
        Handles LineString and MultiLineString.
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
        Computes _base_road_layers: for each elevation layer, the list of SVG paths
        for the road segments inside that layer only (no overlaps between layers).
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
        Draws the contour of for a given elevation range.
        If a border is given for the map, draws the inside of the border

        :param level_range:
        :return:
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
        Draws all the contour layers at different altitude levels

        :param level_steps:
        :return:
        """

        self._base_layers = {}

        for idx, _ in enumerate(level_steps):
            if idx == len(level_steps) - 1:
                break

            print(f"Processing Level {level_steps[idx]}m")
            level_range = (level_steps[idx - 1], level_steps[-1])
            self.compute_base_layer(level_range)

        self.compute_road_layers()

    @property
    def name(self):
        return self._name

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
            for feature in geojson['features']:
                self._borders_polygons.append(shape(feature['geometry']))

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
            "upper_left": pixel2coord(self.gt, 0, 0),
            "upper_right": pixel2coord(self.gt, xsize, 0),
            "lower_left": pixel2coord(self.gt, 0, ysize),
            "lower_right": pixel2coord(self.gt, xsize, ysize),
            "center": pixel2coord(self.gt, xsize // 2, ysize // 2)
        }

        return corners
