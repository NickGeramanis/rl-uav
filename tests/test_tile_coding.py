import numpy as np
import pytest

from rl_uav.features.tile_coding import TileCoding


class TestTileCoding:
    @pytest.fixture(autouse=True)
    def init_tile_coding(self):
        n_tiles_per_dimension = np.array([2, 2])
        displacement_vector = np.array([1, 1])
        n_tilings = 2
        n_actions = 2
        state_space_range = (np.array([0, 0]), np.array([10, 5]))
        self.tile_coding = TileCoding(n_actions,
                                      n_tilings,
                                      n_tiles_per_dimension,
                                      state_space_range)

    def test_get_features(self):
        state = np.array([3, 2])

        features = self.tile_coding.get_features(state, 0)
        expected_features = [1, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 1, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0,
                             0, 0, 0]

        assert (features == expected_features).all()

    def test_calculate_q(self):
        state = np.array([3, 2])
        weights = np.array([i for i in range(36)])

        q = self.tile_coding.calculate_q(weights, state)
        expected_q = [13, 49]

        assert (q == expected_q).all()
