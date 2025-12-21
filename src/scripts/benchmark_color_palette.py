import os.path
from argparse import ArgumentError

from data_models.map import Map
from defines.layer_ranges import LayerRanges
from src.utils.parser import argv_parser
from defines.color_palettes import ColorPalettes
from defines.road_weights import RoadsWeight


def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file
    border_data = args.borders_geojson
    roads_data = args.roads_geojson

    if not os.path.exists(out_data):
        os.makedirs(out_data)

    steps = LayerRanges.linear_15

    contour_map = Map(tif_file=tif_data, borders_geojson=border_data, roads_geojson=roads_data)
    contour_map.road_level = 0x8A
    contour_map.road_scaling = RoadsWeight.RANKING_1
    contour_map.compute_all_layers(level_steps=steps)
    contour_map.show_roads = False
    out_data = os.path.join("/tmp/Maps/color_testing/")

    for palette_name, palette in ColorPalettes.get_all_palettes().items():
        contour_map.name = palette_name
        contour_map.color_palette = palette
        contour_map.save_all_layers(save_path=out_data, combined=True, for_cut=False, remove_inters=True)


if __name__ == "__main__":
    main()
