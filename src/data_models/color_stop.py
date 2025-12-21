from typing import List, Tuple, Union

import numpy as np
from pydantic_extra_types.color import Color


class ColorStop:
    stops: List[Tuple[Union[float, int], np.ndarray[int]]] = None

    def __init__(self, stops: List[Tuple[Union[float, int], Color]]):
        self.stops = []
        for stop in stops:
            weight, color = stop
            nd_a_color = np.array(color.as_rgb_tuple())
            self.stops.append((weight, nd_a_color))
