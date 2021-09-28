from abc import ABC, abstractmethod
from typing import List, Tuple

from rl_uav.spaces.space import Space


class Env(ABC):
    @abstractmethod
    def step(self, action: int) -> Tuple[List[float], float, bool, List[str]]:
        pass

    @abstractmethod
    def reset(self) -> List[float]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def seed(self) -> List[float]:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass
