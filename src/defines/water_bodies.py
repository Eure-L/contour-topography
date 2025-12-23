from enum import Enum


class WaterBodyType(Enum):
    DAM = "DAM"
    RIVER = "RIVER"
    POND = "POND"
    SWAMP = "SWAMP"
    CREEK = "CREEK"
    LAKE = "LAKE"

    def __str__(self):
        return self.value