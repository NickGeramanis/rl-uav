from typing import Any, Tuple, Union

import numpy as np

from rl_uav.spaces.space import Space


class Box(Space):
    """A (possibly unbounded) box in R^n.

    Specifically, a Box represents the Cartesian product
    of n closed intervals. Each interval has the form of one of
    [a, b], (-oo, b], [a, oo), or (-oo, oo).
    There are two common use cases:
    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]),
                high=np.array([2.0, 4.0]),
                dtype=np.float32)
        Box(2,)
    """
    __dtype: np.dtype
    __shape: Tuple[int, ...]
    __low: np.ndarray
    __high: np.ndarray
    __bounded_below: np.ndarray
    __bounded_above: np.ndarray

    def __init__(self,
                 low: Union[float, np.ndarray],
                 high: Union[float, np.ndarray],
                 shape: Tuple = None,
                 dtype: np.dtype = np.float32) -> None:
        self.__dtype = dtype

        # determine shape if it isn't provided directly
        if shape is not None:
            if isinstance(low, np.ndarray) and low.shape != shape:
                raise ValueError("low.shape doesn't match provided shape")
            if isinstance(high, np.ndarray) and high.shape != shape:
                raise ValueError("high.shape doesn't match provided shape")
        elif isinstance(low, np.ndarray):
            shape = low.shape
            if not np.isscalar(high) and high.shape != shape:
                raise ValueError("high.shape doesn't match provided shape")
        elif not np.isscalar(high):
            shape = high.shape
            if not np.isscalar(low) and low.shape != shape:
                raise ValueError("low.shape doesn't match provided shape")
        else:
            raise ValueError('shape must be provided or inferred'
                             'from the shapes of low or high')

        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self.__shape = shape
        self.__low = low
        self.__high = high

        self.__low = self.low.astype(self.dtype)
        self.__high = self.high.astype(self.dtype)

        self.__bounded_below = -np.inf < self.low
        self.__bounded_above = np.inf > self.high

    def is_bounded(self, manner: str = 'both') -> bool:
        below = np.all(self.__bounded_below)
        above = np.all(self.__bounded_above)
        if manner == 'both':
            return below and above
        elif manner == 'below':
            return below
        elif manner == 'above':
            return above
        else:
            raise ValueError('manner is not in {"below", "above", "both"}')

    def sample(self) -> np.ndarray:
        """Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate
        is sampled according to the form of the interval:
        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.__high if self.__dtype.kind == 'f' else (
            self.__high.astype('int64') + 1)
        sample = np.empty(self.__shape)

        # Masking arrays which classify the coordinates
        # according to interval type
        unbounded = ~self.__bounded_below & ~self.__bounded_above
        upp_bounded = ~self.__bounded_below & self.__bounded_above
        low_bounded = self.__bounded_below & ~self.__bounded_above
        bounded = self.__bounded_below & self.__bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = np.random.normal(
            size=unbounded[unbounded].shape)

        sample[low_bounded] = np.random.exponential(
            size=low_bounded[low_bounded].shape) + self.low[low_bounded]

        sample[upp_bounded] = -np.random.exponential(
            size=upp_bounded[upp_bounded].shape) + self.high[upp_bounded]

        sample[bounded] = np.random.uniform(low=self.low[bounded],
                                            high=high[bounded],
                                            size=bounded[bounded].shape)
        if self.__dtype.kind == 'i':
            sample = np.floor(sample)

        return sample.astype(self.__dtype)

    def contains(self, x: np.ndarray) -> bool:
        return (isinstance(x, np.ndarray)
                and x.shape == self.__shape
                and np.all(x >= self.__low)
                and np.all(x <= self.__high))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__shape

    @property
    def dtype(self) -> np.dtype:
        return self.__dtype

    @property
    def low(self) -> np.ndarray:
        return self.__low

    @property
    def high(self) -> np.ndarray:
        return self.__high

    @property
    def bounded_below(self) -> np.ndarray:
        return self.__bounded_below

    @property
    def bounded_above(self) -> np.ndarray:
        return self.__bounded_above

    def __str__(self) -> str:
        return (f'Box({self.__low.min()}, '
                f'{self.__high.max()}, '
                f'{self.__shape}, '
                f'{self.__dtype})')

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Box)
                and self.__shape == other.shape
                and np.allclose(self.__low, other.low)
                and np.allclose(self.__high, other.high))
