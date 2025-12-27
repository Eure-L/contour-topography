import json
import logging
import math
import os
import time
from typing import Dict, Tuple, Union, List
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

from data_models.color_stop import ColorStop
from data_models.features import RoadFeature, WaterFeature
from data_models.features.line_feature import LineFeature
from defines.canvas_sizes import A3
from defines.color_palettes import ColorPalettes
from defines.road_detail import RoadDetail
from defines.road_weights import RoadsWeight
from defines.water_bodies import WaterBodyType
from src.utils.colormapping import altitudes_to_rgb_array, altitude_to_rgb
from utils.colormapping import altitude_to_gray
from utils.geo import pixel2coord, geo_to_pixel, scale_path_y
from utils.inkscape import parallel_convert_strokes_to_paths, batch_rotate_svg

from utils.svg import convert_strokes_to_paths_in_svg, parallel_convert_strokes_to_paths_in_svg

logger = logging.getLogger('map')
logger.setLevel(logging.DEBUG)


class Map:
    """
    Interfaces TIF Image file
    """

    _grayscale_picture: np.ndarray = None
    _border_mask: np.ndarray = None
    _color_picture: np.ndarray = None

    _topo_layers: Dict = None
    _road_layers: Dict[Tuple[int, int], List[Tuple[int, str]]] = None
    _lf_layers: Dict[Tuple[int, int], List[str]] = None
    _water_layers: Dict[Tuple[int, int], List[str]] = None

    _width: int = None
    _height: int = None

    # Data source files
    _tif_file: str = None
    _borders_geojson: str = None
    _roads_geojson: str = None
    _waters_geojson: str = None
    _line_features_geojsons: List[str] = None

    # Deserialized data
    _borders_polygons: List = None
    _road_features: List[RoadFeature] = None
    _water_features: List[WaterFeature] = None
    _line_features: List[LineFeature] = None

    _ds: Dataset = None

    _gt: GeomTransformer = None
    _corners: Dict = None
    _bounding_box: Dict = None
    _name: str = None
    _color_palette: ColorStop = None

    # Public properties
    show_contour_strokes = False
    show_roads = True
    show_water_surfaces = True
    road_detail: RoadDetail = RoadDetail.MEDIUM
    road_scaling = RoadsWeight.RANKING_1
    canevas = A3
    for_cut = False
    combined_grayscale_cut = True
    always_stroke_to_paths = False

    # Only for display purposes, CNC Machine sees it as a vector
    cut_width_mm = 1
    rotate = 0

    filtered_water_bodies: List[WaterBodyType] = []
    size_filtered_water_bodies: List[WaterBodyType] = []
    waters_min_size = 500

    def __init__(self, tif_file: str,
                 borders_geojson: str = None,
                 roads_geojson: str = None,
                 name: str = None,
                 waters_geojson: str = None,
                 line_features_geojsons: List[str] = None):
        """

        :param tif_file:            Tif file storing grayscale values
        :param borders_geojson:     optionnal Geojson describing all the borders of the given Map
        :param name:                Optionnal, Sets the name of the map for file saving
        """
        self._tif_file = tif_file
        self._topo_layers = {}

        if name is not None:
            self._name = name
        else:
            name = os.path.split(tif_file)[1]
            self._name = ''.join(name.split('.')[:-1])

        self._borders_geojson = borders_geojson
        self._roads_geojson = roads_geojson
        self._waters_geojson = waters_geojson
        self._line_features_geojsons = line_features_geojsons

    def elevation_at(self, lon: float, lat: float) -> Union[float, None]:
        """
        Get elevation value at specific geographic coordinates.

        Args:
            lon: Longitude coordinate
            lat: Latitude coordinate

        Returns:
            Elevation value in meters, or None if coordinates are out of bounds
        """
        px, py = geo_to_pixel(self.gt, lon, lat)

        if 0 <= px < self.width and 0 <= py < self.height:
            return float(self.grayscale_picture[py, px])

        return None

    def feature_in_elevation(self, feature: Union[LineString, Polygon, BaseGeometry],
                             level_range: Tuple[float, float]) -> bool:
        """
        Check if any part of a feature (line or polygon) lies within the specified elevation range.

        Args:
            feature: Shapely geometry object (LineString, Polygon, etc.)
            level_range: Tuple of (min_elevation, max_elevation)

        Returns:
            True if any part of the feature is within elevation range, False otherwise
        """
        min_alt, max_alt = level_range

        def check_coords(coords):
            """Helper function to check if any coordinate is within elevation range"""
            for lon, lat in coords:
                elev = self.elevation_at(lon, lat)
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

    def save_layer_as_svg(self, contour: np.ndarray, layer_range: Tuple[Union[int, float], Union[int, float]],
                          save_file: str):
        """
        Save a contour layer as an SVG file.

        Args:
            contour: Numpy array of contour points
            layer_range: Elevation range tuple (min, max)
            save_file: Output SVG file path
        """
        min_alt = self.grayscale_picture.min()
        max_alt = self.grayscale_picture.max()

        if not self.for_cut:
            r, g, b = altitude_to_rgb(layer_range[0], min_alt, max_alt, self.color_palette)
            svg_color = f"rgb({r},{g},{b})"
            self.save_map_as_svgs(contour, save_file,
                                  fill_color=svg_color,
                                  stroke_color=svg_color if not self.show_contour_strokes else "black",
                                  fill=True)
        else:
            if self.combined_grayscale_cut:
                gray_value = 0xff - altitude_to_gray(layer_range[0], min_alt, max_alt)
                svg_color = f"rgb({gray_value},{gray_value},{gray_value})"
                fill = True
            else:
                svg_color = "red"
                fill = False

            self.save_map_as_svgs(contour, save_file,
                                  fill_color=svg_color,
                                  stroke_color="black" if self.show_contour_strokes else svg_color,
                                  fill=fill)

    def save_map_as_svgs(self, contours: np.ndarray, filename: str, fill: bool,
                         fill_color: str = "black",
                         stroke_color: str = "black"):
        """
        Save contours as SVG file with proper scaling and viewbox.

        Args:
            contours: List of contour arrays
            filename: Output SVG file path
            fill: Whether to fill the contours
            fill_color: Stroke/fill color
            stroke_color: Color of the stroke
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

            path = ET.SubElement(root, "ns0:path", type="road", stroke="black", fill="none",
                                 **{"stroke-width": f"{thickness}mm"},
                                 d=d)
            path.tail = "\n  "

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

    def append_lfs_to_svg(self, svg_file: str, lf_paths: List[str]):
        """
        Append generic line features paths to an existing SVG file.

        Args:
            svg_file: Path to existing SVG file
            lf_paths: List of line path (svg_path_data)
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

    def append_water_to_svg(self, svg_file: str, water_paths: List[str]):
        """
        Append water surfaces to an existing SVG file.

        Args:
            svg_file: Path to existing SVG file
            water_paths: List of SVG path data strings for water surfaces
        """
        tree = ET.parse(svg_file)
        root = tree.getroot()

        fill = "blue" if self.for_cut else "#ADD8E6"

        for d in water_paths:
            path = ET.SubElement(root, "ns0:path", type="water", stroke="none", fill=fill,
                                 **{"stroke-width": "0.1mm"}, d=d)
            path.tail = "\n  "

        tree.write(svg_file, encoding="utf-8", xml_declaration=True)

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

    def save_all_layers(self, save_path: str, combined: bool, remove_inters=False):
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
        for level_range, contour in self._topo_layers.items():
            start, top = level_range
            start, top = int(start), int(top)
            file = os.path.join(save_path, f"{self.name}_{start}-{top}.svg")
            saved_layers.append(file)
            self.save_layer_as_svg(contour, (start, top), file)

            if self.show_roads:
                layer_roads = self._road_layers.get(level_range, [])
                if layer_roads:
                    self.append_roads_to_svg(file, layer_roads)

                layer_lf = self._lf_layers.get(level_range, [])
                if layer_lf:
                    self.append_lfs_to_svg(file, layer_lf)

            if self.show_water_surfaces:
                layer_waters = self._water_layers.get(level_range, [])
                if layer_waters:
                    self.append_water_to_svg(file, layer_waters)

        # converts all road strokes to path using inkscape
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

        # rotate SVGs if needed for CNC machine
        if self.rotate != 0:
            batch_rotate_svg(saved_layers, saved_layers, self.rotate)

    def compute_all_layers(self, level_steps: List[int]):
        """
        Compute all elevation layers based on given level steps.

        Args:
            level_steps: List of elevation values defining layer boundaries
        """

        self._topo_layers = {}

        for idx, _ in enumerate(level_steps):
            if idx == len(level_steps) - 1:
                break

            print(f"Processing Level {level_steps[idx]}m")
            level_range = (level_steps[idx - 1], level_steps[-1])
            self.compute_base_layer(level_range)

        if self.show_roads:
            self.compute_road_layers()
            self.compute_lf_layers()

        if self.show_water_surfaces:
            self.compute_water_surfaces()

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

        self._topo_layers[level_range] = contours

    def compute_road_layers(self):
        """
        Compute road layers for each elevation layer.

        Populates self._road_layers with road segments that fall within
        each elevation layer's range.
        """
        self._road_layers = {lr: [] for lr in self._topo_layers.keys()}
        level_ranges = list(self._topo_layers.keys())
        next_level_ranges = level_ranges[1:]
        next_level_ranges.append(level_ranges[-1])

        for idx, level_range in enumerate(level_ranges):
            start, end = level_range
            nex_start, nex_end = next_level_ranges[idx]

            for road in self.roads:
                if road.hierarchy > self.road_detail.value:
                    continue

                if self.feature_in_elevation(road.geometry, (start, nex_start)):
                    for svg_path in road.paths:
                        self._road_layers[level_range].append((road.hierarchy, svg_path))

    def compute_lf_layers(self):
        """
        Compute generic line features layers for each elevation layer.

        Populates self._road_layers with road segments that fall within
        each elevation layer's range.
        """
        self._lf_layers = {lr: [] for lr in self._topo_layers.keys()}
        level_ranges = list(self._topo_layers.keys())
        next_level_ranges = level_ranges[1:]
        next_level_ranges.append(level_ranges[-1])

        for idx, level_range in enumerate(level_ranges):
            start, end = level_range
            nex_start, nex_end = next_level_ranges[idx]

            for lf in self.line_features:
                if self.feature_in_elevation(lf.geometry, (start, nex_start)):
                    for svg_path in lf.paths:
                        self._lf_layers[level_range].append(svg_path)

    def compute_water_surfaces(self):
        """
        Compute water surfaces for each elevation layer.
        If a water body spans multiple layers (e.g., flowing river), it is included in all relevant layers.

        Populates self._water_layers with water surfaces that fall within
        each elevation layer's range.
        """
        self._water_layers = {lr: [] for lr in self._topo_layers.keys()}
        level_ranges = list(self._topo_layers.keys())
        next_level_ranges = level_ranges[1:]
        next_level_ranges.append(level_ranges[-1])

        for idx, level_range in enumerate(level_ranges):
            start, end = level_range
            nex_start, nex_end = next_level_ranges[idx]

            for water in self.water_surfaces:
                # Check if water feature intersects with this elevation range
                if self.feature_in_elevation(water.geometry, (start, nex_start)):
                    # Convert water geometry to SVG paths
                    svg_paths = water.to_svg_paths()
                    self._water_layers[level_range].extend(svg_paths)

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
    def roads(self) -> List[RoadFeature]:
        if self._road_features is None:
            self._road_features = []
            if self._roads_geojson is None:
                return self._road_features

            with open(self._roads_geojson, 'r') as f:
                geojson = json.load(f)

            for feature in geojson['features']:
                road = RoadFeature(feature, gt=self.gt, lat_scale=self.lat_scale, lon_scale=1)
                self._road_features.append(road)

        return self._road_features

    @property
    def line_features(self) -> List[LineFeature]:
        if self._line_features is None:
            self._line_features = []
            if self._line_features_geojsons is None:
                return self._line_features

            for line_features_geojson in self._line_features_geojsons:
                with open(line_features_geojson, 'r') as f:
                    geojson = json.load(f)

                for feature in geojson['features']:
                    line_feature = LineFeature(feature, gt=self.gt, lat_scale=self.lat_scale, lon_scale=1)
                    self._line_features.append(line_feature)

        return self._line_features

    @property
    def water_surfaces(self) -> List[WaterFeature]:
        if self._water_features is None:
            self._water_features = []
            if self._waters_geojson is None:
                return self._water_features

            with open(self._waters_geojson, 'r') as f:
                geojson = json.load(f)

            for feature in geojson['features']:

                # Aplies filters on water bodies to include (lots of swamps and ponds..)
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

                feat = WaterFeature(feature, gt=self.gt, lat_scale=self.lat_scale, lon_scale=1)
                self._water_features.append(feat)

        return self._water_features

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
            self._grayscale_picture = cv2.imread(self._tif_file, cv2.IMREAD_UNCHANGED)
        return self._grayscale_picture

    @property
    def color_picture(self):
        if self._color_picture is None:
            self._color_picture = altitudes_to_rgb_array(self.grayscale_picture)
        return self._color_picture

    @property
    def file(self) -> str:
        return self._tif_file

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
            "upper_left": pixel2coord(self.gt, 0, 0),
            "upper_right": pixel2coord(self.gt, xsize, 0),
            "lower_left": pixel2coord(self.gt, 0, ysize),
            "lower_right": pixel2coord(self.gt, xsize, ysize),
            "center": pixel2coord(self.gt, xsize // 2, ysize // 2)
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
        px1, py1 = geo_to_pixel(self.gt,
                                self.corners["center"][0],
                                self.corners["center"][1]
                                )

        px2, py2 = geo_to_pixel(self.gt,
                                self.corners["center"][0],
                                self.corners["center"][1] + 0.01
                                )

        print("Δpy:", py2 - py1)
