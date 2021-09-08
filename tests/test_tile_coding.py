import numpy as np

from src.features.tile_coding import TileCoding


class TestTileCoding:

    def test_get_features(self):
        low = np.array([-1.2, -0.07], dtype=np.float32)
        high = np.array([0.6, 0.07], dtype=np.float32)
        observation_space = Box(
            low=low, high=high, shape=(2,), dtype=np.float32)
        tiles_per_dimension = [3, 3]
        displacement_vector = [1, 1]
        n_tilings = 2
        n_actions = 2
        tile_coding = TileCoding(
            n_actions, n_tilings, tiles_per_dimension,
            observation_space, displacement_vector)

        state = [-0.5, 0]
        action = 0
        features = tile_coding.get_features(state, action)
        expected_features = np.array(
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        assert np.array_equal(expected_features, features)