import os.path

from src.map import Map


def main():
    TIF_DATA = os.path.join("../data/canberra.tif")
    OUT_DATA = os.path.join("../data/generated2")

    canberra_map = Map(TIF_DATA)
    canberra_map.show_colour_picture()
    # canberra_map.draw_all_layers(50, save_path=OUT_DATA, color=True)

if __name__ == "__main__":
    main()
