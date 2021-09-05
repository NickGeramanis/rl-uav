import numpy as np

from feature_constructor import FeatureConstructor


class TileCoding(FeatureConstructor):

    def __init__(self, n_actions, n_tilings, tiles_per_dimension,
                 observation_space, displacement_vector):
        self.n_tilings = n_tilings
        self.n_actions = n_actions
        self.tiles_per_dimension = np.array(tiles_per_dimension) + 1
        self.n_dimensions = len(self.tiles_per_dimension)
        self.tiles_per_tiling = np.prod(self.tiles_per_dimension)
        self.n_tiles = self.n_tilings * self.tiles_per_tiling

        self.create_tilings(observation_space, displacement_vector)


    def create_tilings(self, observation_space, displacement_vector,):
        tile_width = ((observation_space.high - observation_space.low)
                      / self.tiles_per_dimension)

        tiling_offset = (np.array(displacement_vector)
                         * tile_width / float(self.n_tilings))

        self.tilings = np.empty((self.n_tilings, self.n_dimensions),
                                dtype=object)

        minimum_value = observation_space.low
        maximum_value = observation_space.high + tile_width

        # create the first tile
        for dimension_i in range(self.n_dimensions):
            self.tilings[0, dimension_i] = np.linspace(
                minimum_value[dimension_i], maximum_value[dimension_i],
                self.tiles_per_dimension[dimension_i] + 1)

        # subtract an offset from the previous tiling to create the rest.
        for tiling_i in range(1, self.n_tilings):
            for dimension_i in range(self.n_dimensions):
                self.tilings[tiling_i, dimension_i] = (
                    self.tilings[tiling_i - 1, dimension_i]
                    - tiling_offset[dimension_i])

    def get_active_features(self, variable):
        indices = np.zeros((self.n_tilings,), object)
        dimensions = np.append(self.n_tilings, self.tiles_per_dimension)

        for tiling_i in range(self.n_tilings):
            index = (tiling_i,)
            for dimension_i in range(self.n_dimensions):
                index += (np.digitize(
                    variable[dimension_i],
                    self.tilings[tiling_i, dimension_i]) - 1,)

            for i in range(len(dimensions)):
                indices[tiling_i] += np.prod(dimensions[i + 1:]) * index[i]

        return tuple(indices)

    def calculate_q(self, weights, state):
        active_features = np.array(self.get_active_features(state))

        q = np.empty((self.n_actions,))
        for action in range(self.n_actions):
            q[action] = np.sum(
                weights[action * self.n_tiles + active_features])
        return q

    def get_features(self, state, action):
        active_features = np.array(self.get_active_features(state))

        features = np.zeros((self.n_features,))
        features[action * self.n_tiles + active_features] = 1
        return features
    
    @property
    def info(self):
        return ('Tile Coding: tilings = {}, tiles per dimension = {}'
                .format(self.n_tilings, self.tiles_per_dimension))
    
    @property
    def n_features(self):
        return self.n_tiles * self.n_actions
