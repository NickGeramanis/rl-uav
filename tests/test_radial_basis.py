import numpy as np
import pytest

from rl_uav.features.radial_basis import RadialBasis


class TestRadialBasis:
    @pytest.fixture(autouse=True)
    def init_rbf(self):
        n_actions = 2
        state_space_range = (np.array([0, 0]), np.array([10, 5]))
        centers_per_dimension = [
            [0.33, 0.67],
            [0.33, 0.67]
        ]
        standard_deviation = 0.1
        self.rbf = RadialBasis(n_actions,
                               state_space_range,
                               centers_per_dimension,
                               standard_deviation)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.rbf.get_features(state, 0)
        expected_features = [1,
                             0.7482635675785652,
                             0.024972002042276155,
                             0.0008333973656066949,
                             2.7813195266616742e-05,
                             0, 0, 0, 0, 0]

        assert (np.isclose(expected_features, features)).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(10)])

        q = self.rbf.calculate_q(weights, state)
        expected_q = [0.80081901652, 9.67130291743]

        assert (np.isclose(expected_q, q)).all()
