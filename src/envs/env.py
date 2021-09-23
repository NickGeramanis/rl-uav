from abc import ABCMeta, abstractmethod, abstractproperty


class Env:
    __metaclass__ = ABCMeta

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def seed(self):
        pass

    @abstractproperty
    def action_space(self):
        pass

    @abstractproperty
    def observation_space(self):
        pass
