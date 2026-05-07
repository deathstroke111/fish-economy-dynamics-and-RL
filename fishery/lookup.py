"""Lookup helpers for SD-style interpolation tables."""

from typing import Sequence, Tuple


Point = Tuple[float, float]


def linear_interpolate(x: float, points: Sequence[Point]) -> float:
    """Piecewise-linear interpolation with endpoint clamping."""

    if not points:
        raise ValueError("points must not be empty")

    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    for left, right in zip(points, points[1:]):
        x0, y0 = left
        x1, y1 = right
        if x0 <= x <= x1:
            if x1 == x0:
                return y1
            ratio = (x - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)

    return points[-1][1]
