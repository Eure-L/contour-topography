import os.path

from data_models.map import Map
from defines.layer_ranges import LayerRanges
from defines.road_detail import RoadDetail
from defines.road_weights import RoadsWeight
from defines.water_bodies import WaterBodyType as WB
from src.utils.parser import argv_parser


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

    # instantiate the MAP object
    map = Map(tif_file=tif_data, borders_geojson=border_geojson, roads_geojson=roads_geojson,
              waters_geojson=waters_geojson)

    # Configure parameters
    map.cut_width_mm = 0.5
    map.road_detail = RoadDetail.HIGH
    map.road_scaling = RoadsWeight.RANKING_1
    map.show_roads = True
    map.show_water_surfaces = True
    map.for_cut = False
    map.show_contour_strokes = False
    map.filtered_water_bodies = [WB.DAM]
    map.size_filtered_water_bodies = [WB.CREEK, WB.POND]
    map.waters_min_size = 20
    map.rotate = 0

    for level_cfg_name, levels in LayerRanges.top_picks.items():
        print(level_cfg_name)
        for layer_range_name, layer_range in levels.items():
            name = f"{level_cfg_name}-{layer_range_name}"
            print(name)
            map.compute_all_layers(level_steps=layer_range)
            map.name = name
            map.save_all_layers(save_path=out_data, combined=True, remove_inters=True)


if __name__ == "__main__":
    main()
