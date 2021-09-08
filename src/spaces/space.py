import numpy as np


class Space(object):
    """Defines the observation and action spaces, so you can
    write generic code that applies to any Env.
    For example, you can choose a random action.
    WARNING - Custom observation & action spaces can inherit
    from the `Space` class. However, most use-cases should be covered
    by the existing space classes (`Box`, `Discrete`).
    Moreover, some implementations of Reinforcement Learning
    algorithms might not handle custom spaces properly.
    Use custom spaces with care.
    """

    def __init__(self, shape=None, dtype=None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = None if dtype is None else np.dtype(dtype)

    def sample(self):
        """Uniformly sample an element of this space."""
        raise NotImplementedError

    def contains(self, x):
        """Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError
