from dataclasses import dataclass


@dataclass
class Euler:
    """Orientation represented with euler angles."""
    roll: float
    pitch: float
    yaw: float