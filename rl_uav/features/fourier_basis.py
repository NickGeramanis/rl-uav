"""This module contains the FourierBasis class."""
import itertools
import math
from typing import Tuple

import numpy as np

from rl_uav.features.linear_function_approximation import \
    LinearFunctionApproximation


class FourierBasis(LinearFunctionApproximation):
    """Construct features using Fourier Basis functions."""
    _order: int
    _state_space_range: Tuple[np.ndarray, np.ndarray]
    _n_functions: int
    _integer_vectors: np.ndarray

    def __init__(self,
                 n_actions: int,
                 order: int,
                 state_space_range: Tuple[np.ndarray, np.ndarray]) -> None:
        self._n_actions = n_actions
        self._order = order
        self._state_space_range = state_space_range
        n_dimensions = len(state_space_range[0])
        self._n_functions = (order + 1) ** n_dimensions
        self._integer_vectors = np.array(list(
            itertools.product(np.arange(order + 1), repeat=n_dimensions)))
        self.n_features = self._n_functions * n_actions

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        norm_state = self._normalize(state,
                                     self._state_space_range[0],
                                     self._state_space_range[1])
        for i in range(self._n_functions):
            cos_term = math.pi * np.dot(norm_state, self._integer_vectors[i])
            features[action * self._n_functions + i] = math.cos(cos_term)

        return features
