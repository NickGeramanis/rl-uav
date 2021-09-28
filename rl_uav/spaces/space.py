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
        """Uniformly sample an element of this space."""
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if x is a valid member of this space."""
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        pass
