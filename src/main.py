import os.path

from src.map import Map


def main():
    TIF_DATA = os.path.join("../data/canberra.tif")
    BORDER_DATA = os.path.join("../data/ACTGOV_BORDER_8764495160505726925.geojson")
    OUT_DATA = os.path.join("../data/generated2")

    canberra_map = Map(tif_file=TIF_DATA, borders_geojson=BORDER_DATA)
    canberra_map.compute_all_layers(level_step=100)
    canberra_map.save_layers(save_path=OUT_DATA, color=True, combined=True, for_cut=False)


if __name__ == "__main__":
    main()
