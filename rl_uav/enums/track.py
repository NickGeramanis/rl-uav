"""This module contains the Track enum."""
from enum import Enum
from typing import Tuple


class Track(Enum):
    """The Track enum."""

    # pylint: disable=invalid-name
    spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]

    # pylint: enable=invalid-name

    def __new__(cls,
                value: int,
                spawn_area: Tuple[Tuple[Tuple[float, float],
                                        Tuple[float, float]], ...] = ()):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.spawn_area = spawn_area
        return obj

    TRACK1 = (1,
              (
                  ((-9, -9), (-9, 9)),
                  ((-9, 9), (9, 9)),
                  ((9, 9), (0, 9)),
                  ((0, 9), (0, 0)),
                  ((0, 0), (-9, 0)),
                  ((-9, 0), (-9, -9))
              )
              )

    TRACK2 = (2,
              (
                  ((-0.2, -0.2), (-3.2, 3.1)),
                  ((-0.2, -9.2), (3.1, 3.1)),
                  ((-9.2, -9.2), (3.1, 12.4)),
                  ((-9.2, 6.1), (12.4, 12.4)),
                  ((6.1, 6.1), (12.4, 3)),
                  ((6.1, 9.2), (3, 3)),
                  ((9.2, 9.2), (3, -3)),
                  ((9.2, 6.1), (-3, -3)),
                  ((6.1, 6.1), (-3, -12.4)),
                  ((6.1, -9.2), (-12.4, -12.4)),
                  ((-9.2, -9.2), (-12.4, -3.2)),
                  ((-9.2, -0.2), (-3.2, -3.2))
              )
              )

    TRACK3 = (3,
              (
                  ((-4.7, 4.6), (-9.3, -9.3)),
                  ((4.6, 4.6), (-9.3, -15.3)),
                  ((4.6, 13.8), (-15.2, -15.2)),
                  ((13.8, 13.8), (-15.2, -9.3)),
                  ((13.8, 20), (-9.3, -9.3)),
                  ((20, 20), (-9.3, -6.1)),
                  ((20, 0), (-6.1, 13.8)),
                  ((-20, 0), (-6.1, 13.8)),
                  ((-20, -20), (-6.1, -9.3)),
                  ((-20, -13.9), (-9.3, -9.3)),
                  ((-13.9, -13.9), (-9.3, -15.2)),
                  ((-13.9, -4.7), (-15.2, -15.2)),
                  ((-4.7, -4.7), (-15.2, -9.3))
              )
              )
