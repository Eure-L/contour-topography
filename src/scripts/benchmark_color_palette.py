import logging
import os.path

from ..defines.road_detail import RoadDetail
from ..defines.color_palettes import ColorPalettes
from ..data_models.map import Map
from ..defines.layer_ranges import CanberraLayerRanges
from ..defines.road_weights import RoadsWeight
from ..utils.logger import set_logger
from ..utils.parser import argv_parser

logger = set_logger(level=logging.INFO)

def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file

    border_geojsons = args.borders_geojson.split(';')
    roads_geojsons = args.roads_geojson.split(';')
    lines_geojsons = args.line_features.split(';')

    # instantiate the MAP object
    map = Map(tif_file=tif_data)
    map.add_border_features_list(border_geojsons)
    map.add_road_features_list(roads_geojsons)
    map.add_line_features_list(lines_geojsons)

    if not os.path.exists(out_data):
        os.makedirs(out_data)

    steps = CanberraLayerRanges.third_13_3

    map.road_detail = RoadDetail.ULTRA
    map.road_scaling = RoadsWeight.RANKING_1
    map.generate_elevation_layers(level_steps=steps)
    out_data = os.path.join("/tmp/Maps/color_testing/")

    for palette_name, palette in ColorPalettes.get_all_palettes().items():
        logger.info(f"Generating Palette sample: {palette_name}")
        map.name = palette_name
        map.color_palette = palette
        map.save_all_layers(save_path=out_data, combined=True, intermediates=False)


if __name__ == "__main__":
    main()
