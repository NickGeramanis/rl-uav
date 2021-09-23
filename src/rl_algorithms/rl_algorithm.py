from abc import ABCMeta, abstractmethod
import logging


class RLAlgorithm(object):
    __metaclass__ = ABCMeta

    def __init__(self, log_filename):
        self.__init_logger(log_filename)

    def __init_logger(self, log_filename):
        self._logger = logging.getLogger(__name__)

        if not self._logger.handlers:
            log_formatter = logging.Formatter(
                fmt='%(asctime)s %(levelname)s %(message)s',
                datefmt='%d-%m-%Y %H:%M:%S')

            file_handler = logging.FileHandler('info.log')
            file_handler.setFormatter(log_formatter)
            self._logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            self._logger.addHandler(console_handler)

            self._logger.setLevel(logging.INFO)

    @abstractmethod
    def train(self, n_episodes):
        pass

    @abstractmethod
    def run(self, n_episodes, render=False):
        pass
