"""This module contains the RLAlgorithm class."""
import logging
from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    """This class defines the interface for the RL algorithms."""
    _LOG_FILENAME = 'rl.log'

    _logger: logging.Logger

    def __init__(self) -> None:
        self._init_logger()

    def _init_logger(self) -> None:
        self._logger = logging.getLogger(__name__)

        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        log_formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s %(message)s',
            datefmt='%d-%m-%Y %H:%M:%S')

        file_handler = logging.FileHandler(self._LOG_FILENAME)
        file_handler.setFormatter(log_formatter)
        self._logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self._logger.addHandler(console_handler)

        self._logger.setLevel(logging.INFO)

    @abstractmethod
    def train(self, n_episodes: int) -> None:
        """Execute the training process for a number of episodes."""

    @abstractmethod
    def run(self, n_episodes: int) -> None:
        """Run the environment and act greedily for a number of episodes."""
