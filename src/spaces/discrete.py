import random
import numpy as np

from space import Space


class Discrete(Space):
    """A discrete space in math{ 0, 1, ..., n-1}.
    Example::
        >>> Discrete(2)
    """

    def __init__(self, n):
        assert n >= 0
        self.n = n
        super(Discrete, self).__init__((), np.int64)

    def sample(self):
        return random.randrange(self.n)

    def contains(self, x):
        if isinstance(x, int):
            # Promote list to array for contains check
            as_int = x
        else:
            return False
        return as_int >= 0 and as_int < self.n

    def __repr__(self):
        return 'Discrete({})'.format(self.n)

    def __eq__(self, other):
        return isinstance(other, Discrete) and self.n == other.n