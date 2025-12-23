import os.path
from argparse import ArgumentError

from data_models.map import Map
from defines.layer_ranges import LayerRanges
from defines.road_detail import RoadDetail
from defines.water_bodies import WaterBodyType as WB
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
        raise ArgumentError(
            "Could not parse Steps list, must be a list of integers separated by ';'.\nExample: 0;50;150;1000;1500 ")

    # instantiate the MAP object
    map = Map(tif_file=tif_data, borders_geojson=border_geojson, roads_geojson=roads_geojson,
              waters_geojson=waters_geojson)

    # Configure parameters
    map.cut_width_mm = 1
    map.road_detail = RoadDetail.HIGH
    map.road_scaling = RoadsWeight.RANKING_1
    map.show_roads = True
    map.show_water_surfaces = True
    map.for_cut = True
    map.show_contour_strokes = False
    map.filtered_water_bodies = [WB.DAM]
    map.size_filtered_water_bodies = [WB.CREEK, WB.POND]
    map.waters_min_size = 30
    map.rotate = 270

    # Compute its layers
    map.compute_all_layers(level_steps=LayerRanges.third_13_bis)

    # Save its layeres
    map.save_all_layers(save_path=out_data, combined=combined)


if __name__ == "__main__":
    main()
