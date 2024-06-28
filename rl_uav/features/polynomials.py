"""This module contains the Polynomials class."""
import itertools

import numpy as np

from rl_uav.features.linear_function_approximation import \
    LinearFunctionApproximation


class Polynomials(LinearFunctionApproximation):
    """Construct features using Polynomials."""
    _order: int
    _n_polynomials: int
    _exponents: np.ndarray

    def __init__(self, n_actions: int, order: int, n_dimensions: int) -> None:
        self._n_actions = n_actions
        self._order = order
        self._n_polynomials = (order + 1) ** n_dimensions
        self._exponents = np.array(list(
            itertools.product(np.arange(order + 1), repeat=n_dimensions)))
        self.n_features = self._n_polynomials * n_actions

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        for i in range(self._n_polynomials):
            prod_terms = np.power(state, self._exponents[i])
            features[action * self._n_polynomials + i] = np.prod(prod_terms)

        return features
