"""This module contains the LinearFunctionApproximation class."""
from abc import ABC, abstractmethod

import numpy as np


class LinearFunctionApproximation(ABC):
    """This class defines the interface for the linear function approximation
    techniques."""

    n_features: int = 0

    _n_actions: int = 0

    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        """Calculate the Q values for each action for a given state."""
        q_values = np.empty((self._n_actions,))
        for action in range(self._n_actions):
            features = self.get_features(state, action)
            q_values[action] = np.dot(features, weights)

        return q_values

    @abstractmethod
    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """Get the feature array for a given state and action."""

    def _normalize(self,
                   value: np.ndarray,
                   min_: np.ndarray,
                   max_: np.ndarray) -> np.ndarray:
        numerator = value - min_
        denominator = max_ - min_
        return numerator / denominator
