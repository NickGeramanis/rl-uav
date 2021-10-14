"""This module contains the RadialBasisFunctions class."""
import itertools
from typing import List, Tuple

import numpy as np

from rl_uav.features.feature_constructor import FeatureConstructor


class RadialBasisFunctions(FeatureConstructor):
    """Construct features using Radial Basis Functions."""
    _n_actions: int
    _state_space_range: Tuple[np.ndarray, np.ndarray]
    _centers: np.ndarray
    _variance: float
    _n_functions: int
    n_features: int

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

    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q_values = np.empty((self._n_actions,))
        for action in range(self._n_actions):
            features = self.get_features(state, action)
            q_values[action] = np.dot(features, weights)

        return q_values

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        norm_state = self._normalize(state)
        features[action * self._n_functions] = 1
        for i in range(1, self._n_functions):
            numerator = np.linalg.norm(norm_state - self._centers[i - 1]) ** 2
            exponent = - numerator / self._variance
            features[action * self._n_functions + i] = np.exp(exponent)

        return features

    def _normalize(self, value: np.ndarray) -> np.ndarray:
        numerator = value - self._state_space_range[0]
        denominator = self._state_space_range[1] - self._state_space_range[0]
        return numerator / denominator

    def __str__(self) -> str:
        return ('Radial Basis Function:'
                f'centers = {self._centers}|variance = {self._variance}')
