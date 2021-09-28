from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    @abstractmethod
    def train(self, n_episodes: int) -> None:
        pass

    @abstractmethod
    def run(self, n_episodes: int, render: bool = False) -> None:
        pass
