from abc import ABCMeta, abstractmethod
import logging


class RLAlgorithm(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            log_formatter = logging.Formatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s')
            file_handler = logging.FileHandler('info.log')
            file_handler.setFormatter(log_formatter)
            self._logger.addHandler(file_handler)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            self._logger.addHandler(console_handler)
            self._logger.setLevel(logging.INFO)

    @abstractmethod
    def train(self, training_episodes):
        pass

    @abstractmethod
    def run(self, episodes, render):
        pass
