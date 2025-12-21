import argparse

from src.defines import defaults


def argv_parser():
    parser = argparse.ArgumentParser(
        description='Generate elevation contour layers from geospatial data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-t', '--tif-file',
        type=str,
        default=None,
        help='Path to the input TIFF file containing elevation data',
        required=True
    )

    parser.add_argument(
        '-b', '--borders-geojson',
        type=str,
        default=None,
        help='Path to the GeoJSON file containing border data',
        required=True
    )

    parser.add_argument(
        '-r', '--roads-geojson',
        type=str,
        default=None,
        help='Path to the GeoJSON file containing roads data',
        required=False
    )

    parser.add_argument(
        '-w', '--ws-geojson',
        type=str,
        default=None,
        help='Path to the GeoJSON file containing water surfaces data',
        required=False
    )

    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=defaults.out_dir,
        help='Directory where output files will be saved'
    )

    parser.add_argument(
        '--level-steps',
        type=str,
        default=None,
        help="Elevation list of steps (in meters), ';' separated",
        required=False
    )

    parser.add_argument(
        '--no-color',
        dest='color',
        action='store_false',
        help='Save layers without color (grayscale)'
    )
    parser.set_defaults(color=True)

    parser.add_argument(
        '--no-combined',
        dest='combined',
        action='store_false',
        help='Do not create combined output file'
    )
    parser.set_defaults(combined=True)

    parser.add_argument(
        '--for-cut',
        action='store_true',
        default=False,
        help='Generate output optimized for cutting/machining'
    )


    parsed_args = parser.parse_args()
    return parsed_args

