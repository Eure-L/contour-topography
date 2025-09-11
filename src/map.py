import os
from xml.etree import ElementTree as ET
from typing import Dict, Tuple, Union
from PIL import Image
import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
import svgutils.transform as st

from src.colormapping import altitudes_to_rgb_array, altitude_to_rgb


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


class Map:
    """
    Interfaces TIF Image file
    """

    _grayscale_picture: np.ndarray = None
    _color_picture: np.ndarray = None
    _layers: Dict = None
    _width: int = None
    _height: int = None
    _file: str = None
    _ds: Dataset = None
    _corners: Dict = None
    _bounding_box: Dict = None
    _name: str = None

    def __init__(self, tif_file: str, name: str = None):
        self._file = tif_file
        self._layers = {}

        if name is not None:
            self._name = name
        else:
            name = os.path.split(tif_file)[1]
            self._name = ''.join(name.split('.')[:-1])

    def show_colour_picture(self):
        img = Image.fromarray(self.color_picture, mode='RGB')
        img.show(self.name)

    def _save_layer_svg(self, contour, layer_range, save_file: str, color, for_cut: bool):
        """

        :param save_file:
        :param color:
        :param combined:
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

    def _save_layer_png(self, contour, layer_range, save_file: str, color):
        """

        :param save_path:
        :param color:
        :param combined:
        :return:
        """
        height, width = self.grayscale_picture.shape

        if color:
            color_img = np.zeros((height, width, 4), dtype=np.uint8)
            min_alt = self.grayscale_picture.min()
            max_alt = self.grayscale_picture.max()
            r, g, b = altitude_to_rgb(layer_range[0], min_alt, max_alt)
            color_bgr = (int(b), int(g), int(r), 255)
            cv2.drawContours(color_img, contour, -1, color_bgr, 1)
            cv2.imwrite(save_file, color_img)

        else:
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            cv2.drawContours(mask, contour, -1, (0, 0, 0, 255), 1)
            cv2.imwrite(save_file, contour)

    def save_layers(self, save_path: str, mode: str, color: bool, combined: bool, for_cut: bool = False):
        """

        :param for_cut:
        :param save_path:
        :param color:
        :param combined:
        :return:
        """

        ALLOWED_EXT = ['png', 'svg']

        if mode not in ALLOWED_EXT:
            raise ValueError(f"Extension {mode} is not handled. Only {str(ALLOWED_EXT)}")

        saved_layers = []
        for (start, top), contour in self._layers.items():
            file = os.path.join(save_path, f"{self.name}_{start}-{top}.{mode}")
            saved_layers.append(file)

            if mode == 'svg':
                self._save_layer_svg(contour, (start, top), file, color, for_cut)

            if mode == 'png':
                self._save_layer_png(contour, (start, top), file, color)

        if combined and mode == 'svg':

            first_layer = saved_layers[0]
            first_svg = st.fromfile(first_layer)
            for next_layer in saved_layers[1:]:
                next_svg = st.fromfile(next_layer)
                first_svg.append(next_svg)
            first_svg.save(os.path.join(save_path, self.name + '.svg'))

        if combined and mode == 'png':
            pass

    def compute_layer(self, level_range: Tuple[Union[float, int]]):
        """
        Draws the contour of the file
        :param level:
        :param save:
        :return:
        """

        mask = np.zeros_like(self.grayscale_picture, dtype=np.uint8)
        mask[(self.grayscale_picture >= level_range[0]) & (self.grayscale_picture < level_range[1])] = 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._layers[level_range] = contours

    def compute_all_layers(self, level_step: Union[int, float]) -> Union[str, None]:
        """
        Draws all the
        :param level_step:
        :param save_path:
        :return:
        """

        self._layers = {}

        min_val = int(self.grayscale_picture.min())
        max_val = int(self.grayscale_picture.max())

        for level in range(min_val, max_val, level_step):
            print(f"Processing Level {level}m")
            self.compute_layer((level, level + level_step))

    @property
    def name(self):
        return self._name

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
        return

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
