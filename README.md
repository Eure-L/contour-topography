# Contour Topography
This project helps with topography contour rendering. 
Requires a geotiff file containing altitudes encoded as grayscales to create a parametric object with parametric rendering options.

## Example

<img src="doc/canberra.svg" alt="Canberra with roads" style="width:200px;"/>


## Usages
```python
from src.map import Map
from utils.roads_weights import RoadsWeight

tif_data =      # path to tif file
borders_data =  # path to Geojson borders file
roads_data =    # path to roads geojson file
save_path =     # out path

contour_map = Map(tif_file=tif_data, borders_geojson=border_data, roads_geojson=roads_data)
contour_map.road_level = 0x8B
contour_map.road_scaling = RoadsWeight.RANKING_1

contour_map.compute_all_layers(level_steps=list(range(556,2000,100)))
contour_map.save_layers(save_path=dst, combined=True, for_cut=False)
```

# Prerequisites
This project needs geotiff to be installed.

**Linux Debian-like**
```shell
sudo apt update
sudo apt install \
    unixodbc unixodbc-dev \
    libblosc-dev \
    libarmadillo-dev \
    libqhull-r8.0 libqhull-dev \
    libxerces-c-dev \
    libgeotiff-dev \
    libaec-dev \
    libnetcdf-dev \
    libcfitsio-dev \
    libhdf5-dev \
    libkml-dev \
    libfyba-dev \
    libspatialite-dev \
    libmysqlclient-dev \
    libfreexl-dev \
    libgeos-dev \
    libproj-dev \
    libhdf4-0 \
    libhdf4-dev
```
Make sure its shared libraries are accessible by the python interpreter.