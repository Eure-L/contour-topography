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
    combined = args.combined
    waters_geojson = args.ws_geojson
    _line_features = args.line_features
    line_features_geojsons = _line_features.split(';')

    # instantiate the MAP object
    map = Map(tif_file=tif_data, borders_geojson=border_geojson, roads_geojson=roads_geojson,
              waters_geojson=waters_geojson, line_features_geojsons=line_features_geojsons)

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
    map.rotate = 270
    map.always_stroke_to_paths = True

    # Compute its layers
    map.compute_all_layers(level_steps=LayerRanges.third_13_3)

    # Save its layeres
    map.save_all_layers(save_path=out_data, combined=combined, remove_inters=True)


if __name__ == "__main__":
    main()
