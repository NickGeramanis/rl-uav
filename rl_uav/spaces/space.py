from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class Space(ABC):
    """Defines the observation and action spaces.

    Using Space we can write generic code that applies to any Env.
    For example, you can choose a random action.
    """

    @abstractmethod
    def sample(self) -> Any:
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        pass
