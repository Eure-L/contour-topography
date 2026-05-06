from pydantic_extra_types.color import Color
from ..data_models.color_stop import ColorStop

class ColorPalettes:
    """
    Class containing all color palettes for elevation mapping.
    """

    WHITE = ColorStop([(1.0, Color("FFFFFF")),
                       (0.0, Color("FFFFFF"))]
                      )

    BROWN_1 = ColorStop([(1.0, Color("0xffffff")),
                         (0.9, Color("0x583101")),
                         (0.4, Color("0x8b5e34")),
                         (0.2, Color("0xd4a276")),
                         (0.0, Color("0xffedd8"))]
    )

    BROWN_2 = ColorStop([(1.0, Color("0xffffff")),
                         (0.9, Color("0x40050f")),
                         (0.4, Color("0x5f280b")),
                         (0.2, Color("0x974c02")),
                         (0.0, Color("0xce9c69"))]
                        )

    BROWN_3 = ColorStop([(1.0, Color("0xffffff")),
                         (0.9, Color("0x583101")),
                         (0.4, Color("0x8b5e34")),
                         (0.2, Color("0xd4a276")),
                         (0.0, Color("0xE6D8C3")),
                         ]
                        )

    BLUE_GREEN = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                            (0.7, Color("0x00ffff")),  # Cyan
                            (0.4, Color("0x00ff00")),  # Green
                            (0.1, Color("0x0000ff")),  # Blue
                            (0.0, Color("0x00008b"))]  # Dark blue (lowest)
                           )

    RED_YELLOW = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                            (0.8, Color("0xff0000")),  # Red
                            (0.5, Color("0xffff00")),  # Yellow
                            (0.2, Color("0xffa500")),  # Orange
                            (0.0, Color("0x8b4513"))]  # Brown (lowest)
                           )

    PURPLE_ORANGE = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                               (0.7, Color("0x800080")),  # Purple
                               (0.4, Color("0xffa500")),  # Orange
                               (0.1, Color("0xff4500")),  # Orange-red
                               (0.0, Color("0x8b0000"))]  # Dark red (lowest)
                              )

    GRAYSCALE = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                           (0.7, Color("0xc0c0c0")),  # Silver
                           (0.4, Color("0x808080")),  # Gray
                           (0.1, Color("0x404040")),  # Dark gray
                           (0.0, Color("0x000000"))]  # Black (lowest)
                          )

    OCEAN_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                             (0.7, Color("0xadd8e6")),  # Light blue
                             (0.4, Color("0x0000ff")),  # Blue
                             (0.1, Color("0x00008b")),  # Dark blue
                             (0.0, Color("0x000033"))]  # Very dark blue (lowest)
                            )

    FIRE_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                            (0.8, Color("0xfff0f0")),  # Light pink
                            (0.5, Color("0xff0000")),  # Red
                            (0.2, Color("0xff4500")),  # Orange-red
                            (0.0, Color("0x8b0000"))]  # Dark red (lowest)
                           )

    FOREST_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                              (0.7, Color("0x90ee90")),  # Light green
                              (0.4, Color("0x228b22")),  # Forest green
                              (0.1, Color("0x006400")),  # Dark green
                              (0.0, Color("0x003300"))]  # Very dark green (lowest)
                             )

    DESERT_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                              (0.7, Color("0xfff8dc")),  # Cornsilk
                              (0.4, Color("0xf4a460")),  # Sandy brown
                              (0.1, Color("0xcd853f")),  # Peru
                              (0.0, Color("0x8b4513"))]  # Saddle brown (lowest)
                             )

    ICE_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                           (0.7, Color("0xe0ffff")),  # Light cyan
                           (0.4, Color("0x00ced1")),  # Dark turquoise
                           (0.1, Color("0x008b8b")),  # Dark cyan
                           (0.0, Color("0x00008b"))]  # Dark blue (lowest)
                          )

    SUNSET_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                              (0.7, Color("0xffd700")),  # Gold
                              (0.4, Color("0xff8c00")),  # Dark orange
                              (0.1, Color("0xff4500")),  # Orange-red
                              (0.0, Color("0x8b0000"))]  # Dark red (lowest)
                             )

    MOONLIGHT_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                                 (0.7, Color("0xf0f8ff")),  # Alice blue
                                 (0.4, Color("0xe6e6fa")),  # Lavender
                                 (0.1, Color("0x483d8b")),  # Dark slate blue
                                 (0.0, Color("0x000000"))]  # Black (lowest)
                                )

    EARTH_THEME = ColorStop([(1.0, Color("0xffffff")),  # White (highest)
                             (0.7, Color("0x98fb98")),  # Pale green
                             (0.4, Color("0x8b4513")),  # Saddle brown
                             (0.1, Color("0x228b22")),  # Forest green
                             (0.0, Color("0x006400"))]  # Dark green (lowest)
                            )

    @classmethod
    def get_all_palettes(cls):
        """
        Returns a dictionary mapping palette names to their ColorStop objects.

        Returns:
            dict: Mapping of palette names to ColorStop objects
        """
        return {
            'WHITE': cls.WHITE,
            'BROWN_1': cls.BROWN_1,
            'BROWN_2': cls.BROWN_2,
            'BROWN_3': cls.BROWN_3,
            'BLUE_GREEN': cls.BLUE_GREEN,
            'RED_YELLOW': cls.RED_YELLOW,
            'PURPLE_ORANGE': cls.PURPLE_ORANGE,
            'GRAYSCALE': cls.GRAYSCALE,
            'OCEAN_THEME': cls.OCEAN_THEME,
            'FIRE_THEME': cls.FIRE_THEME,
            'FOREST_THEME': cls.FOREST_THEME,
            'DESERT_THEME': cls.DESERT_THEME,
            'ICE_THEME': cls.ICE_THEME,
            'SUNSET_THEME': cls.SUNSET_THEME,
            'MOONLIGHT_THEME': cls.MOONLIGHT_THEME,
            'EARTH_THEME': cls.EARTH_THEME
        }
    # Reversed GRAYSCALE_WHITE_TO_BLACK_LIGHT palette
    # Reversed GRAYSCALE_WHITE_TO_BLACK_LIGHT palette
    GRAYSCALE_WHITE_TO_BLACK_LIGHT = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                              (1-0.8, Color("0xf0f0f0")),  # Very light gray
                                              (1-0.6, Color("0xd0d0d0")),  # Light gray
                                              (1-0.4, Color("0xa0a0a0")),  # Medium gray
                                              (1-0.2, Color("0x707070")),  # Dark gray
                                              (1-0.0, Color("0x000000"))]  # Black (lowest)
                                             )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_MEDIUM palette
    GRAYSCALE_WHITE_TO_BLACK_MEDIUM = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                                (1-0.75, Color("0xe0e0e0")),  # Light gray
                                                (1-0.5, Color("0xb0b0b0")),  # Medium gray
                                                (1-0.25, Color("0x505050")),  # Dark gray
                                                (1-0.0, Color("0x000000"))]  # Black (lowest)
                                               )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_DARK palette
    GRAYSCALE_WHITE_TO_BLACK_DARK = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                              (1-0.7, Color("0xd0d0d0")),  # Light gray
                                              (1-0.4, Color("0x707070")),  # Dark gray
                                              (1-0.0, Color("0x000000"))]  # Black (lowest)
                                             )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_SOFT palette
    GRAYSCALE_WHITE_TO_BLACK_SOFT = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                             (1-0.8, Color("0xf5f5f5")),  # Off-white
                                             (1-0.6, Color("0xe0e0e0")),  # Light gray
                                             (1-0.4, Color("0xc0c0c0")),  # Silver
                                             (1-0.2, Color("0x808080")),  # Gray
                                             (1-0.0, Color("0x000000"))]  # Black (lowest)
                                            )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_HARD palette
    GRAYSCALE_WHITE_TO_BLACK_HARD = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                              (1-0.9, Color("0xf0f0f0")),  # Very light gray
                                              (1-0.7, Color("0xb0b0b0")),  # Medium gray
                                              (1-0.4, Color("0x505050")),  # Dark gray
                                              (1-0.0, Color("0x000000"))]  # Black (lowest)
                                             )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_CONTRAST palette
    GRAYSCALE_WHITE_TO_BLACK_CONTRAST = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                                  (1-0.9, Color("0xe0e0e0")),  # Light gray
                                                  (1-0.7, Color("0x808080")),  # Gray
                                                  (1-0.4, Color("0x303030")),  # Dark gray
                                                  (1-0.0, Color("0x000000"))]  # Black (lowest)
                                                 )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_PASTEL palette
    GRAYSCALE_WHITE_TO_BLACK_PASTEL = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                                (1-0.8, Color("0xf0f0f0")),  # Very light gray
                                                (1-0.6, Color("0xd0d0d0")),  # Light gray
                                                (1-0.4, Color("0xa0a0a0")),  # Medium gray
                                                (1-0.2, Color("0x707070")),  # Dark gray
                                                (1-0.0, Color("0x000000"))]  # Black (lowest)
                                               )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_METALLIC palette
    GRAYSCALE_WHITE_TO_BLACK_METALLIC = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                                  (1-0.8, Color("0xe0e0e0")),  # Light gray
                                                  (1-0.6, Color("0xc0c0c0")),  # Silver
                                                  (1-0.4, Color("0x808080")),  # Gray
                                                  (1-0.2, Color("0x404040")),  # Dark gray
                                                  (1-0.0, Color("0x000000"))]  # Black (lowest)
                                                 )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_NEUTRAL palette
    GRAYSCALE_WHITE_TO_BLACK_NEUTRAL = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                                 (1-0.75, Color("0xf0f0f0")),  # Very light gray
                                                 (1-0.5, Color("0xc0c0c0")),  # Silver
                                                 (1-0.25, Color("0x606060")),  # Dark gray
                                                 (1-0.0, Color("0x000000"))]  # Black (lowest)
                                                )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_SUBTLE palette
    GRAYSCALE_WHITE_TO_BLACK_SUBTLE = ColorStop([(1-1.0, Color("0xffffff")),  # White (highest)
                                                (1-0.8, Color("0xf5f5f5")),  # Off-white
                                                (1-0.6, Color("0xe0e0e0")),  # Light gray
                                                (1-0.4, Color("0xc0c0c0")),  # Silver
                                                (1-0.2, Color("0x808080")),  # Gray
                                                (1-0.0, Color("0x000000"))]  # Black (lowest)
                                               )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_DEEP palette
    GRAYSCALE_WHITE_TO_BLACK_DEEP = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                             (1-0.7, Color("0xe0e0e0")),  # Light gray
                                             (1-0.4, Color("0x707070")),  # Dark gray
                                             (1-0.0, Color("0x000000"))]  # Black (lowest)
                                            )

    # Reversed GRAYSCALE_WHITE_TO_BLACK_VIBRANT palette
    GRAYSCALE_WHITE_TO_BLACK_VIBRANT = ColorStop([(0.0, Color("0xffffff")),  # White (highest)
                                                 (1-0.8, Color("0xf0f0f0")),  # Very light gray
                                                 (1-0.6, Color("0xd0d0d0")),  # Light gray
                                                 (1-0.4, Color("0xa0a0a0")),  # Medium gray
                                                 (1-0.2, Color("0x707070")),  # Dark gray
                                                 (1-0.0, Color("0x000000"))]  # Black (lowest)
                                                )

    @classmethod
    def get_all_white_to_black_palettes(cls):
        """
        Returns a dictionary mapping white-to-black palette names to their ColorStop objects.

        Returns:
            dict: Mapping of white-to-black palette names to ColorStop objects
        """
        return {
            'GRAYSCALE_WHITE_TO_BLACK_LIGHT': cls.GRAYSCALE_WHITE_TO_BLACK_LIGHT,
            'GRAYSCALE_WHITE_TO_BLACK_MEDIUM': cls.GRAYSCALE_WHITE_TO_BLACK_MEDIUM,
            'GRAYSCALE_WHITE_TO_BLACK_DARK': cls.GRAYSCALE_WHITE_TO_BLACK_DARK,
            'GRAYSCALE_WHITE_TO_BLACK_SOFT': cls.GRAYSCALE_WHITE_TO_BLACK_SOFT,
            'GRAYSCALE_WHITE_TO_BLACK_HARD': cls.GRAYSCALE_WHITE_TO_BLACK_HARD,
            'GRAYSCALE_WHITE_TO_BLACK_CONTRAST': cls.GRAYSCALE_WHITE_TO_BLACK_CONTRAST,
            'GRAYSCALE_WHITE_TO_BLACK_PASTEL': cls.GRAYSCALE_WHITE_TO_BLACK_PASTEL,
            'GRAYSCALE_WHITE_TO_BLACK_METALLIC': cls.GRAYSCALE_WHITE_TO_BLACK_METALLIC,
            'GRAYSCALE_WHITE_TO_BLACK_NEUTRAL': cls.GRAYSCALE_WHITE_TO_BLACK_NEUTRAL,
            'GRAYSCALE_WHITE_TO_BLACK_SUBTLE': cls.GRAYSCALE_WHITE_TO_BLACK_SUBTLE,
            'GRAYSCALE_WHITE_TO_BLACK_DEEP': cls.GRAYSCALE_WHITE_TO_BLACK_DEEP,
            'GRAYSCALE_WHITE_TO_BLACK_VIBRANT': cls.GRAYSCALE_WHITE_TO_BLACK_VIBRANT
        }

# For backward compatibility
ALL_PALETES = [
    ColorPalettes.WHITE,
    ColorPalettes.BROWN_1,
    ColorPalettes.BROWN_2,
    ColorPalettes.BLUE_GREEN,
    ColorPalettes.RED_YELLOW,
    ColorPalettes.PURPLE_ORANGE,
    ColorPalettes.GRAYSCALE,
    ColorPalettes.OCEAN_THEME,
    ColorPalettes.FIRE_THEME,
    ColorPalettes.FOREST_THEME,
    ColorPalettes.DESERT_THEME,
    ColorPalettes.ICE_THEME,
    ColorPalettes.SUNSET_THEME,
    ColorPalettes.MOONLIGHT_THEME,
    ColorPalettes.EARTH_THEME
]