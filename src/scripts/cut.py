import logging

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

    border_geojsons = args.borders_geojson.split(';')
    roads_geojsons = args.roads_geojson.split(';')
    waters_geojsons = args.ws_geojson.split(';')
    lines_geojsons = args.line_features.split(';')

    # instantiate the MAP object
    map = Map(tif_file=tif_data)
    map.add_border_features_list(border_geojsons)
    map.add_road_features_list(roads_geojsons)
    map.add_water_surface_features_list(waters_geojsons)
    map.add_line_features_list(lines_geojsons)

    # Configure parameters
    map.cut_width_mm = 1
    map.road_detail = RoadDetail.ULTRA
    map.road_scaling = RoadsWeight.RANKING_1
    map.for_cut = True
    map.show_contour_lines = False
    map.always_stroke_to_paths = True
    map.filtered_water_bodies = [WB.DAM]
    map.size_filtered_water_bodies = [WB.CREEK, WB.POND]
    map.waters_min_size = 30
    map.rotate = 270

    # Compute its layers
    map.generate_elevation_layers(level_steps=CanberraLayerRanges.third_9_3)

    # Save its layeres
    map.save_all_layers(save_path=out_data, combined=combined, intermediates=True)

if __name__ == "__main__":
    main()
