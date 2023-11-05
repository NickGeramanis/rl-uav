"""This module contains the Action enum."""
from enum import Enum


class Action(Enum):
    """The Action enum."""

    linear_velocity: float
    angular_velocity: float

    def __new__(cls,
                value: int,
                linear_velocity: float = 0,
                angular_velocity: float = 0):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.linear_velocity = linear_velocity
        obj.angular_velocity = angular_velocity
        return obj

    FORWARD = (0, 0.5, 0)
    ROTATE_RIGHT = (1, 0.1, 0.5)
    ROTATE_LEFT = (2, 0.1, -0.5)
