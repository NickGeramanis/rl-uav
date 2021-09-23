from typing import List, Tuple

import numpy as np

from src.features.feature_constructor import FeatureConstructor


class TileCoding(FeatureConstructor):
    __n_tilings = None
    __n_actions = None
    __n_tiles_per_dimension = None
    __n_dimensions = None
    __n_tiles = None
    __tilings = None
    __n_features = None

    def __init__(self, n_actions, n_tilings, n_tiles_per_dimension,
                 state_space_low, state_space_high, displacement_vector):
        self.__n_tilings = n_tilings
        self.__n_actions = n_actions
        self.__n_tiles_per_dimension = np.array(n_tiles_per_dimension) + 1
        self.__n_dimensions = len(self.__n_tiles_per_dimension)
        n_tiles_per_tiling = np.prod(self.__n_tiles_per_dimension)
        self.__n_tiles = n_tilings * n_tiles_per_tiling
        self.__tilings = self.__create_tilings(state_space_low,
                                               state_space_high,
                                               displacement_vector)
        self.__n_features = self.__n_tiles * n_actions

    def __create_tilings(self, state_space_low, state_space_high,
                         displacement_vector):
        width = state_space_high - state_space_low
        tile_width = width / self.__n_tiles_per_dimension
        tiling_offset = displacement_vector * tile_width / self.__n_tilings

        tilings = np.empty((self.__n_tilings, self.__n_dimensions),
                           dtype=np.ndarray)

        min_value = state_space_low
        max_value = state_space_high + tile_width

        # Create the first tile
        for i in range(self.__n_dimensions):
            tilings[0, i] = np.linspace(
                min_value[i], max_value[i],
                num=self.__n_tiles_per_dimension[i] + 1)

        # In order to create the rest tilings,
        # subtract the tiling offset from the previous tiling.
        for i in range(1, self.__n_tilings):
            for j in range(self.__n_dimensions):
                tilings[i, j] = tilings[i - 1, j] - tiling_offset[j]

        return tilings

    def __get_active_features(self, state):
        active_features = np.zeros((self.__n_tilings,), dtype=np.ndarray)
        dimensions = np.append(self.__n_tilings, self.__n_tiles_per_dimension)

        for tiling_i in range(self.__n_tilings):
            index = (tiling_i,)
            for i in range(self.__n_dimensions):
                index += (
                    np.digitize(state[i], self.__tilings[tiling_i, i]) - 1,)

            for j in range(len(dimensions)):
                active_features[tiling_i] += (np.prod(dimensions[j + 1:])
                                              * index[j])

        return tuple(active_features)

    def calculate_q(self, weights, state):
        q = np.empty((self.__n_actions,))
        active_features = self.__get_active_features(state)
        for action in range(self.__n_actions):
            q[action] = np.sum(
                weights[action * self.__n_tiles + active_features])

        return q

    def get_features(self, state, action):
        features = np.zeros((self.n_features,))
        active_features = self.__get_active_features(state)
        features[action * self.__n_tiles + active_features] = 1
        return features

    @property
    def n_features(self):
        return self.__n_features

    def __str__(self):
        return 'Tile Coding: tilings = {}'.format(self.__tilings)
