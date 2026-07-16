import json
import logging
import math
import os
import time
import traceback
from typing import Dict, Tuple, Union, List, Optional
from xml.etree import ElementTree as ET, ElementTree

import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
from osgeo.ogr import GeomTransformer
from shapely import vectorized
from shapely.geometry import shape
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import transform as shp_transform

from utils.svg import save_svg
from .features import RoadFeature, WaterFeature
from .features.line_feature import LineFeature
from ..defines.canvas_sizes import A3
from ..defines.color_palettes import ColorPalettes
from ..defines.road_detail import RoadDetail
from ..defines.road_weights import RoadsWeight
from ..defines.water_bodies import WaterBodyType
from ..utils.colormapping import altitude_to_gray
from ..utils.colormapping import altitudes_to_rgb_array, altitude_to_rgb
from ..utils.geo import pixel2coord, scale_path_y
from ..utils.inkscape import parallel_convert_strokes_to_paths, batch_rotate_svg

# Constants
DEFAULT_MIN_CONTOUR_POINTS = 20
DEFAULT_WATER_MIN_SIZE = 500
DEFAULT_CUT_WIDTH_MM = 1.0
DEFAULT_ROTATE_DEGREES = 0
DEFAULT_PAELTTE = ColorPalettes.BROWN_1
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
ET.register_namespace("inkscape", INKSCAPE_NS)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Map:
    """
    Interfaces TIF Image file and provides functionality to process and visualize elevation data.
    """
    svg_tree_layers: Dict[Tuple[int, int], ElementTree]
    svg_terrain_group: ET.Element
    svg_road_group: ET.Element
    svg_line_group: ET.Element
    svg_water_group: ET.Element

    def __init__(self, tif_file: str, name: Optional[str] = None):
        """
        Initialize the Map object.

        :param tif_file: Path to the TIF file storing grayscale elevation values
        :param name: Optional name for the map (used for file saving)
        """
        self._tif_file = tif_file
        self._name = name or os.path.splitext(os.path.basename(tif_file))[0]

        # Configuration properties
        self.show_contour_lines = False
        self.road_detail = RoadDetail.MEDIUM
        self.road_scaling = RoadsWeight.RANKING_1
        self.canevas = A3
        self.for_cut = False
        self.combined_grayscale_cut = False
        self.always_stroke_to_paths = False
        self.stack_hint = True
        self.cut_width_mm = DEFAULT_CUT_WIDTH_MM
        self.rotate = DEFAULT_ROTATE_DEGREES
        self.color_palette = DEFAULT_PAELTTE

        # Water body filtering
        self.filtered_water_bodies: List[WaterBodyType] = []
        self.size_filtered_water_bodies: List[WaterBodyType] = []
        self.waters_min_size = DEFAULT_WATER_MIN_SIZE

        # Initialize data structures
        self._initialize_data_structures()

    def reset_svg_elements(self):
        """ Initializes SVG rendering elements """
        self.svg_tree: ET.ElementTree = ET.ElementTree(ET.Element("svg", **self.get_svg_header()))
        root = self.svg_tree.getroot()

        self.svg_terrain_group = self._create_svg_layer(root, "terrain", "Terrain")
        self.svg_water_group = self._create_svg_layer(root, "waters", "Water")
        self.svg_road_group = self._create_svg_layer(root, "roads", "Roads")
        self.svg_line_group = self._create_svg_layer(root, "lines", "Lines")

        self.svg_tree_layers: Dict[Tuple[int, int], ET.ElementTree] = {}

    def _initialize_data_structures(self):
        """Initialize all data structures used by the Map class."""
        self._grayscale_picture: Optional[np.ndarray] = None
        self._border_mask: Optional[np.ndarray] = None
        self._color_picture: Optional[np.ndarray] = None

        self._topo_layers: Dict = {}
        self._road_layers: Dict[Tuple[int, int], List[RoadFeature]] = {}
        self._lf_layers: Dict[Tuple[int, int], List[LineFeature]] = {}
        self._water_layers: Dict[Tuple[int, int], List[WaterFeature]] = {}

        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._corners = None

        # Data source files
        self._border_sources: List[str] = []
        self._road_sources: List[str] = []
        self._water_sources: List[str] = []
        self._line_sources: List[str] = []

        # Deserialized data
        self._borders_polygons: List = []
        self._road_features: List[RoadFeature] = []
        self._water_features: List[WaterFeature] = []
        self._line_features: List[LineFeature] = []

        # GDAL objects
        self._ds: Optional[Dataset] = None
        self._gt: Optional[GeomTransformer] = None
        self._corners: Optional[Dict] = None
        self._bounding_box: Optional[Dict] = None

        self.done_roads = set()

        self.reset_svg_elements()

    def _create_svg_layer(self, parent, layer_id: str, label: str):
        return ET.SubElement(
            parent,
            "g",
            {
                "id": layer_id,
                f"{{{INKSCAPE_NS}}}label": label,
                f"{{{INKSCAPE_NS}}}groupmode": "layer",
            }
        )

    # Property getters and setters
    @property
    def name(self) -> str:
        """Get the name of the map."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the name of the map."""
        self._name = value

    @property
    def border_mask(self) -> np.ndarray:
        """Get the border mask array."""
        if self._border_mask is None:
            self._border_mask = self._get_border_mask()
        return self._border_mask

    @property
    def borders_polygons(self) -> List:
        """Get the list of border polygons."""
        if not self._borders_polygons:
            self._load_borders_polygons()
        return self._borders_polygons

    @property
    def roads(self) -> List[RoadFeature]:
        """Get the list of road features."""
        if not self._road_features:
            self._load_road_features()
        return self._road_features

    @property
    def line_features(self) -> List[LineFeature]:
        """Get the list of line features."""
        if not self._line_features:
            self._load_line_features()
        return self._line_features

    @property
    def water_surfaces(self) -> List[WaterFeature]:
        """Get the list of water surface features."""
        if not self._water_features:
            self._load_water_features()
        return self._water_features

    @property
    def width(self) -> int:
        """Get the width of the grayscale picture."""
        if self._width is None:
            _, self._width = self.grayscale_picture.shape
        return self._width

    @property
    def height(self) -> int:
        """Get the height of the grayscale picture."""
        if self._height is None:
            self._height, _ = self.grayscale_picture.shape
        return self._height

    @property
    def grayscale_picture(self) -> np.ndarray:
        """Get the grayscale picture array."""
        if self._grayscale_picture is None:
            self._grayscale_picture = cv2.imread(self._tif_file, cv2.IMREAD_UNCHANGED)
            if self._grayscale_picture is None:
                raise FileNotFoundError(f"Could not read TIF file: {self._tif_file}")
        return self._grayscale_picture

    @property
    def color_picture(self) -> np.ndarray:
        """Get the color picture array."""
        if self._color_picture is None:
            self._color_picture = altitudes_to_rgb_array(self.grayscale_picture)
        return self._color_picture

    @property
    def file(self) -> str:
        """Get the path to the TIF file."""
        return self._tif_file

    @property
    def ds(self) -> Dataset:
        """Get the GDAL dataset."""
        if self._ds is None:
            self._ds = gdal.Open(self.file, gdal.GA_ReadOnly)
            if self._ds is None:
                raise RuntimeError(f"Failed to open GDAL dataset: {self.file}")
        return self._ds

    @property
    def gt(self) -> GeomTransformer:
        """Get the geotransform object."""
        if self._gt is None:
            self._gt = self.ds.GetGeoTransform()
        return self._gt

    @property
    def corners(self) -> Dict[str, Tuple[float, float]]:
        """Get the geographic coordinates of the map corners."""
        if self._corners is None:
            self._corners = self._get_corners()
        return self._corners

    @property
    def level_ranges(self) -> List[Tuple[int, int]]:
        """Get the list of elevation level ranges."""
        return list(self._topo_layers.keys())

    @property
    def bounding_box(self) -> Dict[str, float]:
        """Get the bounding box coordinates of the map."""
        if self._bounding_box is None:
            self._bounding_box = {
                "north_latitude": self.corners["upper_left"][1],
                "south_latitude": self.corners["lower_left"][1],
                "west_longitude": self.corners["upper_left"][0],
                "east_longitude": self.corners["upper_right"][0],
            }

        return self._bounding_box

    @property
    def north_latitude(self) -> float:
        """Get the northern latitude boundary."""
        return self.bounding_box["north_latitude"]

    @property
    def south_latitude(self) -> float:
        """Get the southern latitude boundary."""
        return self.bounding_box["south_latitude"]

    @property
    def east_longitude(self) -> float:
        """Get the eastern longitude boundary."""
        return self.bounding_box["east_longitude"]

    @property
    def west_longitude(self) -> float:
        """Get the western longitude boundary."""
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

    @property
    def lat_scale(self) -> float:
        """
        Scale factor to compensate for EPSG:4326 latitude distortion.
        """
        lat0 = self.corners["center"][1]  # degrees
        scale = 1.0 / math.cos(math.radians(lat0))
        return scale

    # Feature addition methods
    def add_border_features(self, file: str):
        """
        Adds a given file to the map's borders sources list.

        :param file: Path to the GeoJSON file containing border data
        """
        self._border_sources.append(file)

    def add_road_features(self, file: str):
        """
        Adds a given file to the map's roads sources list.

        :param file: Path to the GeoJSON file containing roads data
        """
        self._road_sources.append(file)

    def add_line_features(self, file: str):
        """
        Adds a given file to the map's generic lines sources list.

        :param file: Path to the GeoJSON file containing line features data
        """
        self._line_sources.append(file)

    def add_water_surface_features(self, file: str):
        """
        Adds a given file to the map's water surfaces sources list.

        :param file: Path to the GeoJSON file containing water surface data
        """
        self._water_sources.append(file)

    def add_border_features_list(self, files: List[str]):
        """
        Adds a list of files to the map's borders sources list.

        :param files: List of paths to GeoJSON files containing border data
        """
        self._border_sources.extend(files)

    def add_road_features_list(self, files: List[str]):
        """
        Adds a list of files to the map's roads sources list.

        :param files: List of paths to GeoJSON files containing roads data
        """
        self._road_sources.extend(files)

    def add_line_features_list(self, files: List[str]):
        """
        Adds a list of files to the map's generic lines sources list.

        :param files: List of paths to GeoJSON files containing line features data
        """
        self._line_sources.extend(files)

    def add_water_surface_features_list(self, files: List[str]):
        """
        Adds a list of files to the map's water surfaces sources list.

        :param files: List of paths to GeoJSON files containing water surface data
        """
        self._water_sources.extend(files)

    def _append_layer_to_svg(self, level_range: Tuple[Union[int, float], Union[int, float]]):
        """
        Save a contour layer as an SVG file.

        :param level_range: Elevation range tuple (min, max)
        """

        stroke_width_mm = round(self.cut_width_mm, 1)
        min_alt = self.grayscale_picture.min()
        max_alt = self.grayscale_picture.max()
        contours: np.ndarray = self._topo_layers[level_range]

        layer_group = self.svg_tree_layers[level_range]

        # Get the layer group below this one
        sorted_level_ranges = sorted(self.svg_tree_layers.keys())
        current_index = sorted_level_ranges.index(level_range)

        if current_index > 0:  # If there is a layer below
            lower_level_range = sorted_level_ranges[current_index - 1]
            layer_group_below = self.svg_tree_layers[lower_level_range]
        else:
            layer_group_below = None

        if not self.for_cut:
            r, g, b = altitude_to_rgb(level_range[0], min_alt, max_alt, self.color_palette)
            svg_color = f"rgb({r},{g},{b})"
            stroke_color = svg_color if not self.show_contour_lines else "black"
            fill_str = f'{svg_color}'
        else:
            if self.combined_grayscale_cut:
                gray_value = 0xff - altitude_to_gray(level_range[0], min_alt, max_alt)
                svg_color = f"rgb({gray_value},{gray_value},{gray_value})"
                fill_str = f'{svg_color}'
            else:
                svg_color = "red"
                fill_str = f"white"
            stroke_color = "black" if self.show_contour_lines else svg_color

        for contour in contours:
            path_data = "M " + " L ".join(
                f"{int(x)},{int(y)}" for x, y in contour[:, 0, :]
            )
            path_data = scale_path_y(path_data, self.lat_scale)
            path_data += " Z"

            new_path = ET.Element("path",
                                  type="terrain",
                                  stroke=f"{stroke_color}",
                                  fill=f"{fill_str}",
                                  **{"stroke-width": f"{stroke_width_mm}mm"}, d=path_data)
            new_path.tail = "\n    "

            if self.stack_hint and layer_group_below:
                below_path = ET.Element("path",
                                        type="hint",
                                        stroke=f"black",
                                        fill="none",
                                        opacity="0.3",
                                        **{"stroke-width": f"{0.4}mm"}, d=path_data)
                below_path.tail = "\n    "
                layer_group_below.getroot().append(below_path)

            layer_group.getroot().insert(0, new_path)
            self.svg_terrain_group.insert(0, new_path)

    def _append_roads_to_svg(self, level_range: Tuple[int, int]):
        """
        Appends to the map SVG root the relevant roads for a level range

        :param level_range: Level range data to insert
        :return: None
        """

        layer_group = self.svg_tree_layers[level_range]
        layer_roads = self._road_layers.get(level_range, [])

        for road in layer_roads:
            paths = road.fmt_paths
            hierarchy = road.hierarchy
            thickness = self.road_scaling.interpolate(hierarchy)
            thickness = round(thickness, 1)
            for d in paths:
                new_path = ET.Element(
                    "path",
                    type="road",
                    **{"stroke-width": f"{thickness}mm"},
                    stroke="black",
                    fill="none",
                    d=d
                )
                new_path.tail = "\n    "
                layer_group.getroot().append(new_path)
                self.svg_road_group.append(new_path)

    def _append_lfs_to_svg(self, level_range: Tuple[int, int]):
        """
        Append generic line features paths to an existing SVG file.

        :param level_range: Level range data to insert
        :return: None
        """

        layer_group = self.svg_tree_layers[level_range]
        layer = self._lf_layers.get(level_range, [])
        for lf in layer:
            paths = lf.fmt_paths
            thickness = 0.8
            for d in paths:
                new_path = ET.Element(
                    "path",
                    type="line",
                    stroke="black",
                    fill="none",
                    **{"stroke-width": f"{thickness}mm"},
                    d=d
                )
                new_path.tail = "\n    "
                layer_group.getroot().append(new_path)
                self.svg_line_group.append(new_path)
        pass

    def _append_water_to_svg(self, level_range: Tuple[int, int]):
        """
        Append water surfaces to an existing SVG file.

        :param level_range: Level range data to insert
        :return: None
        """

        layer_group = self.svg_tree_layers[level_range]
        layer = self._water_layers.get(level_range, [])
        fill = "blue" if self.for_cut else "#ADD8E6"

        for wb_feat in layer:
            paths = wb_feat.fmt_paths
            for d in paths:
                new_path = ET.Element(
                    "path",
                    type="water",
                    stroke="none",
                    fill=fill,
                    **{"stroke-width": "0.1mm"},
                    d=d
                )
                new_path.tail = "\n    "
                layer_group.getroot().append(new_path)
                self.svg_water_group.append(new_path)
        pass

    def _get_border_mask(self) -> np.ndarray:
        """
        Create a mask where pixels inside borders are 255, outside are 0.

        :return: Numpy array representing the border mask
        """

        height, width = self.grayscale_picture.shape

        if not self.borders_polygons:
            return np.ones((height, width), dtype=np.uint8) * 255

        xs = np.arange(width)
        ys = np.arange(height)
        xx, yy = np.meshgrid(xs, ys)
        gt = self.gt

        # Pixel -> lon/lat
        x = gt[0] + xx * gt[1]
        y = gt[3] + yy * gt[5]

        # Apply latitude scale
        y = y * self.lat_scale

        combined_inside = np.zeros_like(x, dtype=bool)
        if isinstance(self.borders_polygons, list) and len(self.borders_polygons) > 1:
            for border_poly in self.borders_polygons:
                multipoly = MultiPolygon(border_poly)
                inside = vectorized.contains(multipoly, x, y)
                combined_inside |= inside
        else:
            multipoly = MultiPolygon(self.borders_polygons)
            combined_inside = vectorized.contains(multipoly, x, y)

        return (combined_inside.astype(np.uint8) * 255)

    def get_svg_header(self):
        height, width = self.grayscale_picture.shape
        viewbox_height = int(height * self.lat_scale)
        viewbox_width = width

        return {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{self.canevas.width}",
            "height": f"{self.canevas.height}",
            "viewBox": f"0 0 {viewbox_width} {viewbox_height}"
        }

    def save_all_layers(self, save_path: str, combined: bool = True, intermediates: bool = False):
        """
        Save all elevation layers as SVG files.

        :param save_path: Directory to save SVG files
        :param combined: Whether to combine all layers into one SVG (default: True)
        :param intermediates: Whether to save intermediate layers
        :return: None
        """
        start_time = time.time()
        layer_times = {}
        total_time = 0

        saved_layers = []
        self.svg_terrain_group.clear()
        self.svg_water_group.clear()
        self.svg_road_group.clear()
        self.svg_line_group.clear()

        os.makedirs(save_path, exist_ok=True)

        for level_range, contour in self._topo_layers.items().__reversed__():
            layer_start_time = time.time()

            logger.debug(f"Processing level range: {level_range}")  # Add this debug log

            start, top = float(level_range[0]), float(level_range[1])

            self._append_layer_to_svg(level_range)
            self._append_roads_to_svg(level_range)
            self._append_lfs_to_svg(level_range)
            self._append_water_to_svg(level_range)

            if intermediates:
                file = os.path.join(save_path, f"{self.name}_{start}-{top}.svg")
                save_svg(self.svg_tree_layers[level_range], file)
                saved_layers.append(file)

            layer_time = time.time() - layer_start_time
            layer_times[level_range] = layer_time
            total_time += layer_time

        # Log layer processing times
        for level_range, layer_time in layer_times.items():
            logger.debug(f"Layer {level_range} processed in {layer_time:.2f} seconds")

        if combined:
            combined_start_time = time.time()

            combined_file = os.path.join(save_path, f"{self.name}.svg")
            save_svg(self.svg_tree, combined_file)
            saved_layers.append(combined_file)

            combined_time = time.time() - combined_start_time
            logger.info(f"Combined SVG created in {combined_time:.2f} seconds")
            total_time += combined_time

        nb_threads = os.cpu_count()
        logger.debug(f"#Threads: {nb_threads}")

        if self.always_stroke_to_paths or self.for_cut:
            stroke_start_time = time.time()

            selectors = ['[type="road"]', '[type="line"]']
            parallel_convert_strokes_to_paths(saved_layers, selectors, max_workers=nb_threads)

            stroke_time = time.time() - stroke_start_time
            logger.info(f"Stroke conversion completed in {stroke_time:.2f} seconds")
            total_time += stroke_time

        if self.rotate != DEFAULT_ROTATE_DEGREES:
            rotate_start_time = time.time()

            batch_rotate_svg(saved_layers, saved_layers, self.rotate, max_workers=nb_threads)

            rotate_time = time.time() - rotate_start_time
            logger.info(f"SVG rotation completed in {rotate_time:.2f} seconds")
            total_time += rotate_time

        return saved_layers

    def generate_elevation_layers(self, level_steps: List[int]):
        """
        Compute all elevation layers based on given level steps.

        :param level_steps: List of elevation values defining layer boundaries
        :return: None
        """
        self._topo_layers = {}

        for idx, _ in enumerate(level_steps):
            if idx == len(level_steps) - 1:
                break

            logger.debug(f"Computing contours {level_steps[idx]}m")
            level_range = (level_steps[idx - 1], level_steps[-1])
            self._generate_elevation_contours(level_range)

            root = ET.Element("svg", **self.get_svg_header())
            self.svg_tree_layers[level_range] = ET.ElementTree(root)

        self._compute_road_layers()
        self._compute_lf_layers()
        self._compute_water_surfaces()

    def _generate_elevation_contours(self, level_range: Tuple[Union[float, int], Union[float, int]]):
        """
        Compute contour for a specific elevation range.

        :param level_range: Tuple of (min_elevation, max_elevation)
        :return: None
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

        self._topo_layers[level_range] = contours

    def _compute_road_layers(self):
        """
        Compute road layers for each elevation layer.
        Populates self._road_layers with road segments that fall within
        each elevation layer's range.

        :return: None
        """

        self._road_layers: Dict[Tuple[int, int], List[RoadFeature]] = {lr: [] for lr in self._topo_layers.keys()}

        for road_feat in self.roads:
            if road_feat.hierarchy > self.road_detail.value:
                continue

            possible_level_ranges = road_feat.get_layer_keys(self.level_ranges)

            for level_range in possible_level_ranges:
                self._road_layers[level_range].append(road_feat)

        # self._optimize_road_layers()

    def _compute_lf_layers(self):
        """
        Compute generic line features layers for each elevation layer.
        Populates self._road_layers with road segments that fall within
        each elevation layer's range.

        :return: None
        """

        self._lf_layers: Dict[Tuple[int, int], List[LineFeature]] = {lr: [] for lr in self._topo_layers.keys()}

        for line_feat in self.line_features:
            possible_level_ranges = line_feat.get_layer_keys(self.level_ranges)
            for level_range in possible_level_ranges:
                self._lf_layers[level_range].append(line_feat)

    def _compute_water_surfaces(self):
        """
        Compute water surfaces for each elevation layer.
        If a water body spans multiple layers (e.g., flowing river), it is included in all relevant layers.
        Populates self._water_layers with water surfaces that fall within
        each elevation layer's range.

        :return: None
        """

        self._water_layers: Dict[Tuple[int, int], List[WaterFeature]] = {lr: [] for lr in self._topo_layers.keys()}
        for water_feat in self.water_surfaces:
            possible_level_ranges = water_feat.get_layer_keys(self.level_ranges)
            for level_range in possible_level_ranges:
                self._water_layers[level_range].append(water_feat)

    # Data loading methods
    def _load_borders_polygons(self) -> None:
        """
        Load and process border polygons features from source files.
        """
        self._borders_polygons = []

        for source_file in self._border_sources:
            try:
                with open(source_file, 'r') as f:
                    geojson = json.load(f)
                for feature in geojson['features']:
                    geom = shape(feature['geometry'])
                    # Scale latitude ONLY
                    geom = shp_transform(lambda lon, lat: (lon, lat * self.lat_scale), geom)
                    self._borders_polygons.append(geom)
            except Exception as e:
                logger.error(f"Error loading border features from {source_file}: {str(e)}")

    def _load_road_features(self) -> None:
        """
        Load and process road features from source files.
        """
        self._road_features = []

        for source_file in self._road_sources:
            try:
                with open(source_file, 'r') as f:
                    geojson: Dict = json.load(f)

                # The Road features object may not at the root of the JSON dictionary
                if 'features' not in geojson:
                    for k, v in geojson.items():
                        if isinstance(v, dict) and 'features' in v.keys():
                            geojson = v
                            break

                for feature in geojson['features']:
                    road = RoadFeature(feature, self.gt, self.grayscale_picture,
                                       lat_scale=self.lat_scale, lon_scale=1)
                    self._road_features.append(road)
            except Exception as e:
                logger.error(f"{traceback.format_exception(e)}")
                logger.error(f"Error loading road features from {source_file}: {str(e)}")

    def _load_line_features(self) -> None:
        """
        Load and process line features from source files.
        """
        self._line_features = []

        for source_file in self._line_sources:
            try:
                with open(source_file, 'r') as f:
                    geojson = json.load(f)

                for feature in geojson['features']:
                    line_feature = LineFeature(feature, self.gt, self.grayscale_picture,
                                               lat_scale=self.lat_scale, lon_scale=1)
                    self._line_features.append(line_feature)
            except Exception as e:
                logger.error(f"Error loading line features from {source_file}: {str(e)}")

    def _load_water_features(self) -> None:
        """
        Load and process water features from source files.
        """
        self._water_features = []

        for source_file in self._water_sources:
            try:
                with open(source_file, 'r') as f:
                    geojson = json.load(f)

                for feature in geojson['features']:
                    # Apply filters on water bodies to include (lots of swamps and ponds..)
                    wb = feature["properties"]['WATER_BODY_TYPE']
                    wb = WaterBodyType(wb)

                    if wb in self.filtered_water_bodies:
                        continue
                    if wb in self.size_filtered_water_bodies:
                        s = len(feature["geometry"]['coordinates'][0])
                        id = feature["properties"]['OBJECTID']
                        wbname = feature["properties"]['WATER_BODY_NAME']
                        if s < self.waters_min_size:
                            continue

                    feat = WaterFeature(feature, self.gt, self.grayscale_picture,
                                        lat_scale=self.lat_scale, lon_scale=1)
                    self._water_features.append(feat)
            except Exception as e:
                logger.error(f"Error loading water features from {source_file}: {str(e)}")

    def get_km_width(self) -> float:
        """
        Calculate the scale of the map in kilometers per pixel, accounting for latitude variations.

        Returns:
            float: The scale in kilometers per pixel
        """
        # Get the center latitude of the map
        center_lat = self.corners["center"][1]
        km_per_degree = 40075 * math.cos(math.radians(center_lat)) / 360
        lon_diff = self.east_longitude - self.west_longitude
        distance_km = lon_diff * km_per_degree

        return distance_km
