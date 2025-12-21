import os.path
from argparse import ArgumentError

from data_models.map import Map
from defines.layer_ranges import LayerRanges
from src.utils.parser import argv_parser
from defines.road_weights import RoadsWeight


def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file
    border_geojson = args.borders_geojson
    roads_geojson = args.roads_geojson
    waters_geojson = args.ws_geojson
    level_steps = args.level_steps
    for_cut = args.for_cut
    combined = args.combined

    if not os.path.exists(out_data):
        os.makedirs(out_data)
    try:
        steps = LayerRanges.linear_15 if not level_steps else [int(step) for step in level_steps.split(';')]
    except Exception as e:
        raise ArgumentError("Could not parse Steps list, must be a list of integers separated by ';'.\nExample: 0;50;150;1000;1500 ")

    # instantiate the MAP object
    contour_map = Map(tif_file=tif_data, borders_geojson=border_geojson, roads_geojson=roads_geojson, waters_geojson=waters_geojson)
    contour_map.road_level = 0xfff
    contour_map.road_scaling = RoadsWeight.RANKING_1
    contour_map.show_roads = True
    contour_map.show_contour_strokes = True

    # Compute its layers
    contour_map.compute_all_layers(level_steps=LayerRanges.third_13)

    # Save its layeres
    contour_map.save_all_layers(save_path=out_data, combined=combined, for_cut=for_cut)




if __name__ == "__main__":
    main()
