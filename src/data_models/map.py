import json
import logging
import math
import os
from typing import Dict, Tuple, Union, List, Optional
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
from osgeo.ogr import GeomTransformer
from shapely import vectorized, Polygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import transform as shp_transform

from data_models.features import RoadFeature, WaterFeature
from data_models.features.line_feature import LineFeature
from defines.canvas_sizes import A3
from defines.color_palettes import ColorPalettes
from defines.road_detail import RoadDetail
from defines.road_weights import RoadsWeight
from defines.water_bodies import WaterBodyType
from src.utils.colormapping import altitudes_to_rgb_array, altitude_to_rgb
from utils.colormapping import altitude_to_gray
from utils.geo import pixel2coord, geo_to_pixel, scale_path_y, elevation_at
from utils.inkscape import parallel_convert_strokes_to_paths, batch_rotate_svg

# Constants
DEFAULT_MIN_CONTOUR_POINTS = 20
DEFAULT_WATER_MIN_SIZE = 500
DEFAULT_CUT_WIDTH_MM = 1.0
DEFAULT_ROTATE_DEGREES = 0
DEFAULT_PAELTTE = ColorPalettes.BROWN_1

logger = logging.getLogger('map')
logger.setLevel(logging.DEBUG)


class Map:
    """
    Interfaces TIF Image file and provides functionality to process and visualize elevation data.
    """

    def __init__(self, tif_file: str, name: Optional[str] = None):
        """
        Initialize the Map object.

        :param tif_file: Path to the TIF file storing grayscale elevation values
        :param name: Optional name for the map (used for file saving)
        """
        self._tif_file = tif_file
        self._name = name or os.path.splitext(os.path.basename(tif_file))[0]

        # Initialize data structures
        self._initialize_data_structures()

        # Configuration properties
        self.show_contour_lines = False
        self.include_roads = True
        self.include_water_surfaces = True
        self.road_detail = RoadDetail.MEDIUM
        self.road_scaling = RoadsWeight.RANKING_1
        self.canevas = A3
        self.for_cut = False
        self.combined_grayscale_cut = True
        self.always_stroke_to_paths = False
        self.cut_width_mm = DEFAULT_CUT_WIDTH_MM
        self.rotate = DEFAULT_ROTATE_DEGREES
        self.color_palette = DEFAULT_PAELTTE

        # Water body filtering
        self.filtered_water_bodies: List[WaterBodyType] = []
        self.size_filtered_water_bodies: List[WaterBodyType] = []
        self.waters_min_size = DEFAULT_WATER_MIN_SIZE

    def _initialize_data_structures(self):
        """Initialize all data structures used by the Map class."""
        self._grayscale_picture: Optional[np.ndarray] = None
        self._border_mask: Optional[np.ndarray] = None
        self._color_picture: Optional[np.ndarray] = None

        self._topo_layers: Dict = {}
        self._road_layers: Dict[Tuple[int, int], List[Tuple[int, str]]] = {}
        self._lf_layers: Dict[Tuple[int, int], List[str]] = {}
        self._water_layers: Dict[Tuple[int, int], List[str]] = {}

        self._width: Optional[int] = None
        self._height: Optional[int] = None

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

    # Core functionality methods
    def feature_in_elevation(self, feature: Union[LineString, Polygon, BaseGeometry],
                          level_range: Tuple[float, float]) -> bool:
        """
        Check if any part of a feature (line or polygon) lies within the specified elevation range.

        :param feature: Shapely geometry object (LineString, Polygon, etc.)
        :param level_range: Tuple of (min_elevation, max_elevation)
        :return: True if any part of the feature is within elevation range, False otherwise
        """
        min_alt, max_alt = level_range

        def check_coords(coords):
            """Helper function to check if any coordinate is within elevation range"""
            for lon, lat in coords:
                elev = elevation_at(self.gt, self.grayscale_picture, lon, lat)
                if elev is not None and min_alt <= elev < max_alt:
                    return True
            return False

        if isinstance(feature, LineString):
            return check_coords(list(feature.coords))
        elif isinstance(feature, Polygon):
            # Check exterior ring
            if check_coords(list(feature.exterior.coords)):
                return True

            # Check interior rings (holes)
            for interior in feature.interiors:
                if check_coords(list(interior.coords)):
                    return True
            return False
        elif hasattr(feature, 'geoms'):  # MultiLineString, MultiPolygon, etc.
            # Check all parts
            for part in feature.geoms:
                if self.feature_in_elevation(part, level_range):
                    return True
            return False
        else:
            return False

    def _save_layer_as_svg(self, contour: np.ndarray, layer_range: Tuple[Union[int, float], Union[int, float]],
                           save_file: str):
        """
        Save a contour layer as an SVG file.

        :param contour: Numpy array of contour points
        :param layer_range: Elevation range tuple (min, max)
        :param save_file: Output SVG file path
        """
        min_alt = self.grayscale_picture.min()
        max_alt = self.grayscale_picture.max()

        if not self.for_cut:
            r, g, b = altitude_to_rgb(layer_range[0], min_alt, max_alt, self.color_palette)
            svg_color = f"rgb({r},{g},{b})"
            self._save_map_as_svgs(contour, save_file,
                                   fill_color=svg_color,
                                   stroke_color=svg_color if not self.show_contour_lines else "black",
                                   fill=True)
        else:
            if self.combined_grayscale_cut:
                gray_value = 0xff - altitude_to_gray(layer_range[0], min_alt, max_alt)
                svg_color = f"rgb({gray_value},{gray_value},{gray_value})"
                fill = True
            else:
                svg_color = "red"
                fill = False

            self._save_map_as_svgs(contour, save_file,
                                   fill_color=svg_color,
                                   stroke_color="black" if self.show_contour_lines else svg_color,
                                   fill=fill)

    def _save_map_as_svgs(self, contours: np.ndarray, filename: str, fill: bool,
                          fill_color: str = "black",
                          stroke_color: str = "black"):
        """
        Save contours as SVG file with proper scaling and viewbox.

        :param contours: List of contour arrays
        :param filename: Output SVG file path
        :param fill: Whether to fill the contours
        :param fill_color: Stroke/fill color
        :param stroke_color: Color of the stroke
        :return: None
        """
        stroke_width_mm = round(self.cut_width_mm, 1)
        height, width = self.grayscale_picture.shape
        viewbox_height = int(height * self.lat_scale)
        viewbox_width = width

        with open(filename, 'w') as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'width="{self.canevas.width}" height="{self.canevas.height}" viewBox="0 0 {viewbox_width} {viewbox_height}">\n')
            for contour in contours:
                path_data = "M " + " L ".join(
                    f"{int(x)},{int(y)}" for x, y in contour[:, 0, :]
                )
                path_data = scale_path_y(path_data, self.lat_scale)
                path_data += " Z"
                fill_str = f'fill="{fill_color}"' if fill else f'fill="none"'
                f.write(
                    f'  <path type="cut" stroke="{stroke_color}" {fill_str} stroke-width="{stroke_width_mm}mm" d="{path_data}" />\n')
            f.write('</svg>')

    def _append_roads_to_svg(self, svg_file: str, road_paths: List[Tuple[int, str]]):
        """
        Append road paths to an existing SVG file.

        :param svg_file: Path to existing SVG file
        :param road_paths: List of tuples (hierarchy, svg_path_data)
        :return: None
        """

        tree = ET.parse(svg_file)
        root = tree.getroot()

        for road in road_paths:
            hierarchy, d = road
            thickness = self.road_scaling.interpolate(hierarchy)
            thickness = round(thickness, 1)

            path = ET.SubElement(root, "ns0:path", type="road", stroke="black", fill="none",
                                 **{"stroke-width": f"{thickness}mm"},
                                 d=d)
            path.tail = "\n  "

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

    def _append_lfs_to_svg(self, svg_file: str, lf_paths: List[str]):
        """
        Append generic line features paths to an existing SVG file.

        :param svg_file: Path to existing SVG file
        :param lf_paths: List of line path (svg_path_data)
        :return: None
        """

        tree = ET.parse(svg_file)
        root = tree.getroot()

        for lf in lf_paths:
            thickness = 0.8

            path = ET.SubElement(root, "ns0:path", type="line_feature", stroke="black", fill="none",
                                 **{"stroke-width": f"{thickness}mm"},
                                 d=lf)
            path.tail = "\n  "

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

    def _append_water_to_svg(self, svg_file: str, water_paths: List[str]):
        """
        Append water surfaces to an existing SVG file.

        :param svg_file: Path to existing SVG file
        :param water_paths: List of SVG path data strings for water surfaces
        :return: None
        """
        tree = ET.parse(svg_file)
        root = tree.getroot()

        fill = "blue" if self.for_cut else "#ADD8E6"

        for d in water_paths:
            path = ET.SubElement(root, "ns0:path", type="water", stroke="none", fill=fill,
                                 **{"stroke-width": "0.1mm"}, d=d)
            path.tail = "\n  "

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

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
        # Pixel → lon/lat
        x = gt[0] + xx * gt[1]
        y = gt[3] + yy * gt[5]
        # Apply latitude scale
        y = y * self.lat_scale

        multipoly = MultiPolygon(self.borders_polygons)
        inside = vectorized.contains(multipoly, x, y)

        return (inside.astype(np.uint8) * 255)

    def save_all_layers(self, save_path: str, combined: bool, remove_inters=False):
        """
        Save all elevation layers as SVG files.

        :param save_path: Directory to save SVG files
        :param combined: Whether to combine all layers into one SVG
        :param remove_inters: Whether to remove intermediary built layers after combining
        :return: None
        """
        saved_layers = []

        os.makedirs(save_path, exist_ok=True)

        # Save each layer as an individual SVG
        for level_range, contour in self._topo_layers.items():
            start, top = int(level_range[0]), int(level_range[1])
            file = os.path.join(save_path, f"{self.name}_{start}-{top}.svg")
            saved_layers.append(file)
            self._save_layer_as_svg(contour, (start, top), file)

            if self.include_roads:
                layer_roads = self._road_layers.get(level_range, [])
                if layer_roads:
                    self._append_roads_to_svg(file, layer_roads)

                layer_lf = self._lf_layers.get(level_range, [])
                if layer_lf:
                    self._append_lfs_to_svg(file, layer_lf)

            if self.include_water_surfaces:
                layer_waters = self._water_layers.get(level_range, [])
                if layer_waters:
                    self._append_water_to_svg(file, layer_waters)

        # Convert all road strokes to paths using inkscape
        if self.always_stroke_to_paths or self.for_cut:
            selectors = ['[type="road"]', '[type="line_feature"]']
            parallel_convert_strokes_to_paths(saved_layers, selectors, max_workers=12)

        # Combine layers into a single SVG if requested
        if combined and saved_layers:
            height, width = self.grayscale_picture.shape
            viewbox_height = int(height * self.lat_scale)
            viewbox_width = width
            combined_svg = ET.Element(
                "svg",
                xmlns="http://www.w3.org/2000/svg",
                width=self.canevas.width,
                height=self.canevas.height,
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
            merged_svg = os.path.join(save_path, f"{self.name}.svg")
            ET.ElementTree(combined_svg).write(merged_svg, encoding="utf-8", xml_declaration=True)
            logger.info(f"Combined SVG saved to {merged_svg}")

            # Remove intermediary layers if requested
            if remove_inters:
                for layer_file in saved_layers:
                    try:
                        os.remove(layer_file)
                        logger.info(f"Removed intermediary layer: {layer_file}")
                    except OSError as e:
                        logger.error(f"Error removing file {layer_file}: {e}")
                saved_layers = []
            saved_layers.append(merged_svg)

        # Rotate SVGs if needed for CNC machine
        if self.rotate != DEFAULT_ROTATE_DEGREES:
            batch_rotate_svg(saved_layers, saved_layers, self.rotate)

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

            print(f"Processing Level {level_steps[idx]}m")
            level_range = (level_steps[idx - 1], level_steps[-1])
            self._generate_elevation_contours(level_range)

        if self.include_roads:
            self._compute_road_layers()
            self._compute_lf_layers()

        if self.include_water_surfaces:
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

        self._road_layers = {lr: [] for lr in self._topo_layers.keys()}

        for road_feat in self.roads:
            if road_feat.hierarchy > self.road_detail.value:
                continue

            possible_level_ranges = road_feat.get_layer_keys(self.level_ranges)
            svg_paths = road_feat.paths

            for level_range in possible_level_ranges:
                for svg_path in svg_paths:
                    self._road_layers[level_range].append((road_feat.hierarchy, svg_path))

    def _compute_lf_layers(self):
        """
        Compute generic line features layers for each elevation layer.
        Populates self._road_layers with road segments that fall within
        each elevation layer's range.

        :return: None
        """

        self._lf_layers = {lr: [] for lr in self._topo_layers.keys()}

        for line_feat in self.line_features:
            possible_level_ranges = line_feat.get_layer_keys(self.level_ranges)
            for level_range in possible_level_ranges:
                svg_paths = line_feat.paths
                self._lf_layers[level_range].extend(svg_paths)

    def _compute_water_surfaces(self):
        """
        Compute water surfaces for each elevation layer.
        If a water body spans multiple layers (e.g., flowing river), it is included in all relevant layers.
        Populates self._water_layers with water surfaces that fall within
        each elevation layer's range.

        :return: None
        """

        self._water_layers = {lr: [] for lr in self._topo_layers.keys()}
        for water_feat in self.water_surfaces:
            possible_level_ranges = water_feat.get_layer_keys(self.level_ranges)
            for level_range in possible_level_ranges:
                svg_paths = water_feat.paths
                self._water_layers[level_range].extend(svg_paths)

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
                    geojson = json.load(f)

                for feature in geojson['features']:
                    road = RoadFeature(feature, self.gt, self.grayscale_picture,
                                     lat_scale=self.lat_scale, lon_scale=1)
                    self._road_features.append(road)
            except Exception as e:
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
