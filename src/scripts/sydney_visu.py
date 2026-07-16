import logging

from defines.layer_ranges import SydneyLayerRanges
from ..defines.color_palettes import ColorPalettes
from ..data_models.map import Map
from ..defines.layer_ranges import CanberraLayerRanges
from ..defines.road_detail import RoadDetail
from ..defines.road_weights import RoadsWeight
from ..defines.water_bodies import WaterBodyType as WB
from ..utils.parser import argv_parser
from ..utils.logger import set_logger

logger = set_logger(level=logging.DEBUG)

def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file
    combined = args.combined

    border_geojsons = args.borders_geojson.split(';') if args.borders_geojson is not None else None
    roads_geojsons = args.roads_geojson.split(';') if args.roads_geojson is not None else None
    waters_geojsons = args.ws_geojson.split(';') if args.ws_geojson is not None else None
    lines_geojsons = args.line_features.split(';') if args.line_features is not None else None

    # instantiate the MAP object
    map = Map(tif_file=tif_data)
    if border_geojsons is not None:
        map.add_border_features_list(border_geojsons)
    if roads_geojsons is not None:
        map.add_road_features_list(roads_geojsons)
    if waters_geojsons is not None:
        map.add_water_surface_features_list(waters_geojsons)
    if lines_geojsons is not None:
        map.add_line_features_list(lines_geojsons)

    # Configure parameters
    map.cut_width_mm = 0.5
    map.road_detail = RoadDetail.LOW
    map.road_scaling = RoadsWeight.RANKING_1
    map.for_cut = False
    map.show_contour_lines = False
    map.filtered_water_bodies = [WB.DAM]
    map.size_filtered_water_bodies = [WB.CREEK, WB.POND]
    map.waters_min_size = 20
    # map.rotate = 270
    map.always_stroke_to_paths = True

    map.color_palette = ColorPalettes.BROWN_4

    # Compute its layers
    map.generate_elevation_layers(level_steps=SydneyLayerRanges._12_layers)

    # Save its layeres
    map.save_all_layers(save_path=out_data, combined=True, intermediates=False)

    logger.info(map.get_km_width())

if __name__ == "__main__":
    main()
