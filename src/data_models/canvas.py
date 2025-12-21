from dataclasses import dataclass


@dataclass
class Canvas:
    width_mm: float
    height_mm: float

    @property
    def aspect_ratio(self):
        return self.width_mm / self.height_mm

    @property
    def width(self) -> str:
        return f"{self.width_mm}mm"

    @property
    def height(self) -> str:
        return f"{self.height_mm}mm"
