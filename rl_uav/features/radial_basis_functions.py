import itertools
from typing import List

import numpy as np

from rl_uav.features.feature_constructor import FeatureConstructor


class RadialBasisFunctions(FeatureConstructor):
    __n_actions: int
    __state_space_low: np.ndarray
    __state_space_high: np.ndarray
    __centers: np.ndarray
    __variance: float
    __n_functions: int
    __n_features: int

    def __init__(self,
                 n_actions: int,
                 state_space_low: np.ndarray,
                 state_space_high: np.ndarray,
                 centers_per_dimension: List[List[float]],
                 standard_deviation: float) -> None:
        self.__n_actions = n_actions
        self.__state_space_low = state_space_low
        self.__state_space_high = state_space_high
        self.__centers = np.array(list(
            itertools.product(*centers_per_dimension)))
        self.__variance = 2 * standard_deviation ** 2
        self.__n_functions = self.__centers.shape[0] + 1
        self.__n_features = self.__n_functions * n_actions

    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q = np.empty((self.__n_actions,))
        for action in range(self.__n_actions):
            features = self.get_features(state, action)
            q[action] = np.dot(features, weights)

        return q

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        norm_state = self.__normalize(state)
        features[action * self.__n_functions] = 1
        for i in range(1, self.__n_functions):
            numerator = np.linalg.norm(norm_state - self.__centers[i - 1]) ** 2
            exponent = - numerator / self.__variance
            features[action * self.__n_functions + i] = np.exp(exponent)

        return features

    def __normalize(self, value: np.ndarray) -> np.ndarray:
        numerator = value - self.__state_space_low
        denominator = self.__state_space_high - self.__state_space_low
        return numerator / denominator

    @property
    def n_features(self) -> int:
        return self.__n_features

    def __str__(self) -> str:
        return ('Radial Basis Function:'
                f'centers = {self.__centers}|variance = {self.__variance}')
