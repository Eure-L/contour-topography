import os.path

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

    contour_map = Map(tif_file=tif_data, borders_geojson=border_data, roads_geojson=roads_data)
    contour_map.road_level = 0x8A
    contour_map.road_scaling = RoadsWeight.RANKING_1
    contour_map.color_palette = ColorPalettes.BROWN_1
    contour_map.show_contour_strokes = True
    contour_map.show_roads = False
    out_data = os.path.join("/tmp/Maps/level_testing/")

    for level_cfg_name, levels in LayerRanges.layer_configs.items():
        print(level_cfg_name)
        for layer_range_name, layer_range in levels.items():
            name = f"{level_cfg_name}-{layer_range_name}"
            print(name)
            contour_map.compute_all_layers(level_steps=layer_range)
            contour_map.name = name
            contour_map.save_all_layers(save_path=out_data, combined=True, for_cut=False, remove_inters=True)

if __name__ == "__main__":
    main()
