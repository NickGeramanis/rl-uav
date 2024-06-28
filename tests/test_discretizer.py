import numpy as np
import pytest

from rl_uav.features.discretizer import Discretizer


class TestDiscretizer:
    @pytest.fixture(autouse=True)
    def init_discretizer(self):
        n_bins = (10, 5)
        state_space_range = (np.array([0, 0]), np.array([10, 5]))
        self.discretizer = Discretizer(n_bins, state_space_range)

    def test_discretize(self):
        state = np.array([3.2, 4.2])

        discrete_state = self.discretizer.discretize(state)

        assert discrete_state == (3, 4)
