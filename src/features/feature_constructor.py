from abc import ABCMeta, abstractmethod, abstractproperty


class FeatureConstructor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_features(self, state, action):
        pass

    @abstractmethod
    def calculate_q(self, weights, state):
        pass

    @abstractproperty
    def info(self):
        pass

    @abstractproperty
    def n_features(self):
        pass
