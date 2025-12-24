from enum import Enum


class RoadDetail(Enum):
    LOW = 0x7
    MEDIUM = 0x8A
    HIGH = 0x8B
    ULTRA = 0xFFF

    def __str__(self):
        return self.value