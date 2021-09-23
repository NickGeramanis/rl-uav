from abc import ABCMeta, abstractmethod, abstractproperty


class FeatureConstructor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_q(self, weights, state):
        pass

    @abstractmethod
    def get_features(self, state, action):
        pass

    @abstractproperty
    def n_features(self):
        pass
