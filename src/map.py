import os
from xml.etree import ElementTree as ET
from typing import Dict, Tuple, Union
from PIL import Image
import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset

from src.colormapping import altitudes_to_rgb_array, altitude_to_rgb


def pixel2coord(gt, px, py):
    x = gt[0] + px * gt[1] + py * gt[2]
    y = gt[3] + px * gt[4] + py * gt[5]
    return (x, y)


def save_contours_as_svg(contours, width, height, filename, color="black"):
    """Save contours as SVG."""
    with open(filename, 'w') as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        for contour in contours:
            path_data = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in contour[:, 0, :])
            path_data += " Z"
            f.write(f'  <path d="{path_data}" stroke="{color}" fill="{color}" stroke-width="1"/>\n')
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
    _layers = None
    _width: int = None
    _height: int = None
    _file: str = None
    _ds: Dataset = None
    _corners: Dict = None
    _bounding_box: Dict = None
    _name: str = None

    def __init__(self, tif_file: str, name: str = None):
        self._file = tif_file
        self._layers = []

        if name is not None:
            self._name = name
        else:
            name = os.path.split(tif_file)[1]
            self._name = ''.join(name.split('.')[:-1])

    def show_colour_picture(self):
        img = Image.fromarray(self.color_picture, mode='RGB')
        img.show(self.name)

    def draw_layer(self, level_range: Tuple[Union[float, int]], save_path: str = None,
                   color: bool = None) -> Union[
        str, None]:
        """
        Draws the contour of the file
        :param level:
        :param save:
        :return:
        """

        mask = np.zeros_like(self.grayscale_picture, dtype=np.uint8)
        mask[(self.grayscale_picture >= level_range[0]) & (self.grayscale_picture < level_range[1])] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = self.grayscale_picture.shape

        if save_path is not None:
            png_file = os.path.join(save_path, f"{self.name}_{level_range[0]}-{level_range[1]}.png")
            svg_file = os.path.join(save_path, f"{self.name}_{level_range[0]}-{level_range[1]}.svg")
            print(f"Saving {png_file}")
            print(f"Saving {svg_file}")

        # Color
        if color is not None:
            color_img = np.zeros((height, width, 3), dtype=np.uint8)
            min_alt = self.grayscale_picture.min()
            max_alt = self.grayscale_picture.max()
            r, g, b = altitude_to_rgb(level_range[0], min_alt, max_alt)
            color_bgr = (int(b), int(g), int(r))
            cv2.drawContours(color_img, contours, -1, color_bgr, 1)
            svg_color = f"rgb({r},{g},{b})"

            # Save to file
            if save_path is None:
                img = Image.fromarray(color_img, mode='RGB')
                img.show(self.name)
                return None

            # Display
            else:
                cv2.imwrite(png_file, color_img)
                save_contours_as_svg(contours, width, height, svg_file, color=svg_color)
                return svg_file

        # Grey scale
        else:

            # Save to file
            if save_path is None:
                img = Image.fromarray(mask, mode='L')
                img.show(self.name)
                return None

            # Display
            else:
                cv2.imwrite(png_file, mask)
                save_contours_as_svg(contours, width, height, svg_file, color="black")
                return svg_file

    def draw_all_layers(self, level_step: Union[int, float], save_path: str = None, color=True) -> Union[str, None]:
        """
        Draws all the
        :param level_step:
        :param save_path:
        :return:
        """

        min_val = int(self.grayscale_picture.min())
        max_val = int(self.grayscale_picture.max())

        for level in range(min_val, max_val, level_step):
            print(f"Processing Level {level}m")
            ret = self.draw_layer((level, level + level_step), save_path, color)
            if ret is not None:
                self._layers.append(ret)

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
