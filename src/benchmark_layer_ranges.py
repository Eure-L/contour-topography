import os.path
from argparse import ArgumentError

from src.map import Map
from src.utils.parser import argv_parser
from utils.colors import ColorPalettes
from utils.roads_weights import RoadsWeight

# Implementation of wood layer configurations for CNC laser cutting

# 13 levels configuration
lowest_13 = [400, 500, 556, 564, 581, 610, 654, 716, 799, 906, 1040, 1204, 1401, 1634, 1900]
second_13 = [400, 500, 556, 595, 635, 688, 755, 837, 936, 1053, 1188, 1343, 1518, 1715, 1900]
third_13 = [400, 500, 556, 625, 702, 787, 880, 981, 1090, 1207, 1332, 1465, 1606, 1755, 1900]
linear_13 = [400, 500, 556, 668, 780, 892, 1004, 1116, 1228, 1340, 1452, 1564, 1676, 1788, 1900]

# 11 levels configuration
lowest_11 = [400, 500, 556, 567, 591, 634, 702, 799, 931, 1102, 1318, 1584, 1900]
second_11 = [400, 500, 556, 602, 655, 726, 819, 936, 1079, 1248, 1446, 1674, 1900]
third_11 = [400, 500, 556, 640, 735, 842, 960, 1090, 1231, 1384, 1549, 1725, 1900]
linear_11 = [400, 500, 556, 690, 825, 959, 1093, 1228, 1362, 1497, 1631, 1765, 1900]

# 9 levels configuration
lowest_9 = [400, 500, 556, 571, 610, 683, 799, 969, 1204, 1513, 1900]
second_9 = [400, 500, 556, 614, 688, 794, 936, 1118, 1343, 1614, 1900]
third_9 = [400, 500, 556, 663, 787, 930, 1090, 1269, 1465, 1680, 1900]
linear_9 = [400, 500, 556, 724, 892, 1060, 1228, 1396, 1564, 1732, 1900]

# Create dictionary with all configurations
layer_configs = {
    'L13': {
        'lowest': lowest_13,
        'second': second_13,
        'third': third_13,
        'linear': linear_13
    },
    'L11': {
        'lowest': lowest_11,
        'second': second_11,
        'third': third_11,
        'linear': linear_11
    },
    'L9': {
        'lowest': lowest_9,
        'second': second_9,
        'third': third_9,
        'linear': linear_9
    }
}


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
    contour_map.show_roads = True

    for level_cfg_name, levels in layer_configs.items():
        print(level_cfg_name)
        for layer_range_name, layer_range in levels.items():
            name = f"{level_cfg_name}-{layer_range_name}"
            print(name)
            contour_map.compute_all_layers(level_steps=layer_range)
            out_data = os.path.join("/tmp/Maps/level_testing/")
            contour_map.name = name
            contour_map.save_layers(save_path=out_data, combined=True, for_cut=False, remove_inters=True)

if __name__ == "__main__":
    main()
