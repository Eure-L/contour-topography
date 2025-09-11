import os.path

from src.map import Map


def main():
    TIF_DATA = os.path.join("../data/canberra.tif")
    OUT_DATA = os.path.join("../data/generated2")

    canberra_map = Map(TIF_DATA)
    canberra_map.compute_all_layers(200)
    canberra_map.save_layers(OUT_DATA, 'svg', True, True, False)

if __name__ == "__main__":
    main()
