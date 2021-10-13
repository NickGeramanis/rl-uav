"""This module contains the RLAlgorithm class."""
from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    """This class defines the interface of the rl algorithms."""

    @abstractmethod
    def train(self, n_episodes: int) -> None:
        """Execute the training process for a number of episodes."""

    @abstractmethod
    def run(self, n_episodes: int, render: bool = False) -> None:
        """Run the environment for a number of episodes without learning."""
