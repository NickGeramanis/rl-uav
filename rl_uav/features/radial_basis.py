"""This module contains the RadialBasis class."""
import itertools
from typing import List, Tuple

import numpy as np

from rl_uav.features.linear_function_approximation import \
    LinearFunctionApproximation


class RadialBasis(LinearFunctionApproximation):
    """Construct features using Radial Basis functions."""
    _state_space_range: Tuple[np.ndarray, np.ndarray]
    _centers: np.ndarray
    _variance: float
    _n_functions: int

    def __init__(self,
                 n_actions: int,
                 state_space_range: Tuple[np.ndarray, np.ndarray],
                 centers_per_dimension: List[List[float]],
                 standard_deviation: float) -> None:
        self._n_actions = n_actions
        self._state_space_range = state_space_range
        self._centers = np.array(list(
            itertools.product(*centers_per_dimension)))
        self._variance = 2 * standard_deviation ** 2
        self._n_functions = self._centers.shape[0] + 1
        self.n_features = self._n_functions * n_actions

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        norm_state = self._normalize(state,
                                     self._state_space_range[0],
                                     self._state_space_range[1])
        features[action * self._n_functions] = 1
        for i in range(1, self._n_functions):
            numerator = np.linalg.norm(norm_state - self._centers[i - 1]) ** 2
            exponent = - numerator / self._variance
            features[action * self._n_functions + i] = np.exp(exponent)

        return features
