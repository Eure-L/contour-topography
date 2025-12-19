from pydantic_extra_types.color import Color

from utils.color_stop import ColorStop

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
