class PathVector:
    dx: int
    dy: int

    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    @property
    def value(self):
        return self.dx, self.dy

    def __sub__(self, other: PathVector):
        return self.dx - other.dx, self.dy - other.dy

    def __eq__(self, other: PathVector):
        substract = self - other
        return substract == (0, 0)
