import numpy as np
import pytest

from rl_uav.features.fourier_basis import FourierBasis


class TestFourierBasis:
    @pytest.fixture(autouse=True)
    def init_fourier_basis(self):
        order = 1
        n_actions = 2
        state_space_range = (np.array([0, 0]), np.array([10, 5]))
        self.fourier_basis = FourierBasis(n_actions, order, state_space_range)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.fourier_basis.get_features(state, 0)
        expected_features = [1,
                             0.30901699437,
                             0.58778525229,
                             -0.58778525229,
                             0, 0, 0, 0]

        assert (np.isclose(expected_features, features)).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(8)])

        q = self.fourier_basis.calculate_q(weights, state)
        expected_q = [-0.27876825793, 4.95729971957]

        assert (np.isclose(expected_q, q)).all()
