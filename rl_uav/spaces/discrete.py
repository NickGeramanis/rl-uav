import random
from typing import Any, Tuple

import numpy as np

from rl_uav.spaces.space import Space


class Discrete(Space):
    """A discrete space in math{0, 1, ..., n-1}."""

    __n: int
    __dtype: np.dtype
    __shape: Tuple[int, ...]

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError('n must not be negative')
        self.__n = n
        self.__dtype = np.int64
        self.__shape = ()

    def sample(self) -> int:
        return random.randrange(self.n)

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.int_)) and 0 <= x < self.__n

    @property
    def n(self) -> int:
        return self.__n

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__shape

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    def __str__(self) -> str:
        return f'Discrete({self.__n})'

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Discrete) and self.__n == other.n
