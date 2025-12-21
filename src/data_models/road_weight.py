from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RoadWeight:
    """
    Interpolates road width based on hierarchy ranking.
    """
    steps: List[Tuple[float, int]]

    def __post_init__(self):
        # Sort steps by hierarchy id (ascending)
        self.steps.sort(key=lambda x: x[1])

    def interpolate(self, hierarchy_id: int) -> float:
        """
        Interpolate the road width for a given hierarchy id.

        :param hierarchy_id: Road hierarchy identifier (int, hex allowed)
        :return: Interpolated width value in [0, 1]
        """
        # Clamp to bounds
        if hierarchy_id <= self.steps[0][1]:
            return self.steps[0][0]

        if hierarchy_id >= self.steps[-1][1]:
            return self.steps[-1][0]

        # Find surrounding control points
        for (w0, h0), (w1, h1) in zip(self.steps, self.steps[1:]):
            if h0 <= hierarchy_id <= h1:
                # Linear interpolation
                t = (hierarchy_id - h0) / (h1 - h0)
                return w0 + t * (w1 - w0)

        # Should never happen
        raise RuntimeError("Interpolation failed")


