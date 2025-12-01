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
from shapely import vectorized
from shapely.geometry.multipolygon import MultiPolygon

from src.utils.colormapping import altitudes_to_rgb_array, altitude_to_rgb
from shapely.geometry import shape, Point
import logging

logger = logging.getLogger('map')
logger.setLevel(logging.DEBUG)


def pixel2coord(gt, px, py):
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return (x, y)


def save_contours_as_svg(contours, width, height, filename, fill: bool, color="black"):
    """Save contours as SVG."""
    with open(filename, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        for contour in contours:
            path_data = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in contour[:, 0, :])
            path_data += " Z"
            fill_str = f'fill="{color}"' if fill else f'fill="none"'
            f.write(f'  <path d="{path_data}" stroke="{color}" {fill_str} stroke-width="1"/>\n')
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
    _layers: Dict = None
    _width: int = None
    _height: int = None
    _file: str = None
    _borders_geojson: str = None
    _borders_polygons: List = None
    _ds: Dataset = None
    _corners: Dict = None
    _bounding_box: Dict = None
    _name: str = None

    def __init__(self, tif_file: str, borders_geojson: str = None, name: str = None):
        """

        :param tif_file:            Tif file storing grayscale values
        :param borders_geojson:     optionnal Geojson describing all the borders of the given Map
        :param name:                Optionnal, Sets the name of the map for file saving
        """
        self._file = tif_file
        self._layers = {}

        if name is not None:
            self._name = name
        else:
            name = os.path.split(tif_file)[1]
            self._name = ''.join(name.split('.')[:-1])

        if borders_geojson is not None:
            self._borders_geojson = borders_geojson

    def show_colour_picture(self):
        img = Image.fromarray(self.color_picture, mode='RGB')
        img.show(self.name)

    def _save_layer_svg(self, contour, layer_range: Tuple[Union[int, float], Union[int, float]], save_file: str, color,
                        for_cut: bool):
        """
        Saves a given layer altitude range as SVG

        :param save_file: dst save file
        :param color:
        :param for_cut:
        :return:
        """
        height, width = self.grayscale_picture.shape

        if not for_cut and color:
            min_alt = self.grayscale_picture.min()
            max_alt = self.grayscale_picture.max()
            r, g, b = altitude_to_rgb(layer_range[0], min_alt, max_alt)
            svg_color = f"rgb({r},{g},{b})"
            save_contours_as_svg(contour, width, height, save_file, color=svg_color, fill=not for_cut)
        else:
            save_contours_as_svg(contour, width, height, save_file, color="black", fill=not for_cut)

    def save_layers(self, save_path: str, color: bool, combined: bool, for_cut: bool = False):
        """
        Saves all computed layers to SVGs or a single SVG.

        :param for_cut:     Wether its for CNC cutting
        :param save_path:   Dst path
        :param color:       Wether to draw the colors according to the altitude
        :param combined:    Wether to combine them all SVGs
        :return:
        """

        saved_layers = []
        for (start, top), contour in self._layers.items():
            start, top = int(start), int(top)
            file = os.path.join(save_path, f"{self.name}_{start}-{top}.svg")
            saved_layers.append(file)

            self._save_layer_svg(contour, (start, top), file, color, for_cut)

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
        gt = self.ds.GetGeoTransform()

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

    def compute_layer(self, level_range: Tuple[Union[float, int], Union[float, int]]):
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

        self._layers[level_range] = contours

    def compute_all_layers(self, for_cut=False, level_steps: List[int] = None):
        """
        Draws all the contour layers at different altitude levels

        :param for_cut:
        :param level_steps:
        :return:
        """

        self._layers = {}

        for idx, _ in enumerate(level_steps):
            if idx == len(level_steps) - 1:
                break

            print(f"Processing Level {level_steps[idx]}m")
            level_range = (0 if for_cut else level_steps[idx-1], level_steps[idx])
            self.compute_layer(level_range)

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
    def width(self):
        if self._width is None:
            _, width = self.grayscale_picture.shape
            self._width = width
        return self.width

    @property
    def height(self):
        if self._height is None:
            _, height = self.grayscale_picture.shape
            self._height = height
        return self.height

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
        gt = self.ds.GetGeoTransform()
        xsize = self.ds.RasterXSize
        ysize = self.ds.RasterYSize

        # Pixel to geo coordinates
        corners = {
            "upper_left": pixel2coord(gt, 0, 0),
            "upper_right": pixel2coord(gt, xsize, 0),
            "lower_left": pixel2coord(gt, 0, ysize),
            "lower_right": pixel2coord(gt, xsize, ysize),
            "center": pixel2coord(gt, xsize // 2, ysize // 2)
        }

        return corners
