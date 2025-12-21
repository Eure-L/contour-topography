from dataclasses import dataclass


@dataclass
class Canvas:
    width_mm: float
    height_mm: float

    @property
    def aspect_ratio(self):
        return self.width_mm / self.height_mm