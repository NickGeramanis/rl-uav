import numpy as np
import pytest

from rl_uav.features.polynomials import Polynomials


class TestPolynomials:
    @pytest.fixture(autouse=True)
    def init_polynomials(self):
        order = 1
        n_actions = 2
        n_dimensions = 2
        self.polynomials = Polynomials(n_actions, order, n_dimensions)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.polynomials.get_features(state, 0)
        expected_features = [1, 2, 3, 6,
                             0, 0, 0, 0]

        assert (features == expected_features).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(8)])

        q = self.polynomials.calculate_q(weights, state)
        expected_q = [26, 74]

        assert (q == expected_q).all()
