from pydantic import BaseModel
from pydantic_extra_types import color


class Line(BaseModel):
    width: int
    color: color.Color


class MachineConfiguration(BaseModel):
    max_width: int
    max_height: int
    cutting_line: Line
    marking_line: Line
