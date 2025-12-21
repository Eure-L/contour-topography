from pydantic_extra_types.color import Color
from data_models.color_stop import ColorStop

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