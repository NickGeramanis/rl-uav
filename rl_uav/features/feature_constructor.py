from abc import ABC, abstractmethod

import numpy as np


class FeatureConstructor(ABC):

    @abstractmethod
    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        pass
