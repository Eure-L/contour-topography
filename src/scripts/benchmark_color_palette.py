import os.path

from data_models.map import Map
from defines.color_palettes import ColorPalettes
from defines.layer_ranges import LayerRanges
from defines.road_weights import RoadsWeight
from src.utils.parser import argv_parser


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

    if not os.path.exists(out_data):
        os.makedirs(out_data)

    steps = LayerRanges.third_13_3

    map.road_detail = 0x8A
    map.road_scaling = RoadsWeight.RANKING_1
    map.generate_elevation_layers(level_steps=steps)
    map.include_roads = False
    out_data = os.path.join("/tmp/Maps/color_testing/")

    for palette_name, palette in ColorPalettes.get_all_palettes().items():
        map.name = palette_name
        map.color_palette = palette
        map.save_all_layers(save_path=out_data, combined=True, remove_inters=True)


if __name__ == "__main__":
    main()
