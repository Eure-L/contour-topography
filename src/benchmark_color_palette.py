import os.path
from argparse import ArgumentError

from src.map import Map
from src.utils.parser import argv_parser
from utils.colors import ColorPalettes
from utils.roads_weights import RoadsWeight


def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file
    border_data = args.borders_geojson
    roads_data = args.roads_geojson
    level_steps = args.level_steps

    if not os.path.exists(out_data):
        os.makedirs(out_data)

    try:
        steps = [int(step) for step in level_steps.split(';')]
    except Exception as e:
        raise ArgumentError(
            "Could not parse Steps list, must be a list of integers sbenchmark_color_palette.pyeparated by ';'.\nExample: 0;50;150;1000;1500 ")

    contour_map = Map(tif_file=tif_data, borders_geojson=border_data, roads_geojson=roads_data)
    contour_map.road_level = 0x8A
    contour_map.road_scaling = RoadsWeight.RANKING_1
    contour_map.compute_all_layers(level_steps=steps)

    for palette_name, palette in ColorPalettes.get_all_palettes().items():
        out_data = os.path.join("/tmp/Maps/color_testing/")
        contour_map.name = palette_name
        contour_map.color_palette = palette
        contour_map.save_layers(save_path=out_data, combined=True, for_cut=False, remove_inters=True)


if __name__ == "__main__":
    main()
