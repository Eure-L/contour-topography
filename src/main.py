import os.path
from mailbox import FormatError

from src.map import Map
from src.utils.parser import argv_parser


def main():
    args = argv_parser()

    out_data = args.output_dir
    tif_data = args.tif_file
    border_data = args.borders_geojson
    roads_data = args.roads_geojson
    level_steps = args.level_steps
    for_cut = args.for_cut
    combined = args.combined
    color = args.color

    if not os.path.exists(out_data):
        os.makedirs(out_data)

    try:
        steps = [int(step) for step in level_steps.split(';')]
    except Exception as e:
        raise FormatError(
            "Could not parse Steps list, must be a list of integers separated by ';'.\nExample: 0;50;150;1000;1500 ")

    contour_map = Map(tif_file=tif_data, borders_geojson=border_data, roads_geojson=roads_data)
    contour_map.compute_all_layers(level_steps=steps)
    contour_map.save_layers(save_path=out_data, color=color, combined=combined, for_cut=for_cut)


if __name__ == "__main__":
    main()
