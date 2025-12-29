import logging

from ..data_models.map import Map
from ..defines.layer_ranges import LayerRanges
from ..defines.road_detail import RoadDetail
from ..defines.road_weights import RoadsWeight
from ..defines.water_bodies import WaterBodyType as WB
from ..utils.logger import set_logger
from ..utils.parser import argv_parser

logger = set_logger(level=logging.INFO)


def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file

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
    map.cut_width_mm = 0.5
    map.road_detail = RoadDetail.HIGH
    map.road_scaling = RoadsWeight.RANKING_1
    map.for_cut = False
    map.show_contour_lines = False
    map.filtered_water_bodies = [WB.DAM]
    map.size_filtered_water_bodies = [WB.CREEK, WB.POND]
    map.waters_min_size = 20
    map.rotate = 0

    for level_cfg_name, levels in LayerRanges.top_picks.items():
        for layer_range_name, layer_range in levels.items():
            name = f"{level_cfg_name}-{layer_range_name}"
            logger.info(f"Generating Layer sample: {name} {layer_range}")
            map.reset_svg_elements()
            map.generate_elevation_layers(level_steps=layer_range)
            map.name = name
            map.save_all_layers(save_path=out_data, combined=True, intermediates=False)


if __name__ == "__main__":
    main()
