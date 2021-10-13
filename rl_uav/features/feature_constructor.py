"""This module contains the FeatureConstructor class."""
from abc import ABC, abstractmethod

import numpy as np


class FeatureConstructor(ABC):
    """This class defines the interface of the feature constructors."""

    @abstractmethod
    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        """Calculate the Q values for each action for a given state."""

    @abstractmethod
    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        """Get the feature array for a given state and action."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """The total number of features."""
