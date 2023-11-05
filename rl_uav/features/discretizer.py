"""This module contains the Discretizer class."""
from typing import Tuple

import numpy as np


# pylint: disable-next=too-few-public-methods
class Discretizer:
    """Discretize the continuous space."""
    n_bins: Tuple[int, ...]
    _bins: np.ndarray

    def __init__(self,
                 n_bins: Tuple[int, ...],
                 state_space_range: Tuple[np.ndarray, np.ndarray]) -> None:
        self.n_bins = n_bins
        n_dimensions = len(n_bins)

        self._bins = np.empty((n_dimensions,), dtype=np.ndarray)
        for i in range(n_dimensions):
            self._bins[i] = np.linspace(state_space_range[0][i],
                                        state_space_range[1][i],
                                        num=n_bins[i] + 1)

    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretize the continuous state."""
        discrete_state = [np.digitize(state[i], bin_) - 1
                          for i, bin_ in enumerate(self._bins)]

        return tuple(discrete_state)
