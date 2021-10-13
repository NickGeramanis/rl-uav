"""This module contains the TileCoding class."""
from typing import Tuple

import numpy as np

from rl_uav.features.feature_constructor import FeatureConstructor


class TileCoding(FeatureConstructor):
    """Construct features using Tile Coding."""
    __n_tilings: int
    __n_actions: int
    __n_tiles_per_dimension: np.ndarray
    __n_dimensions: int
    __n_tiles: int
    __tilings: np.ndarray
    __n_features: int

    def __init__(self,
                 n_actions: int,
                 n_tilings: int,
                 n_tiles_per_dimension: np.ndarray,
                 state_space_range: Tuple[np.ndarray, np.ndarray]) -> None:
        self.__n_tilings = n_tilings
        self.__n_actions = n_actions
        self.__n_tiles_per_dimension = n_tiles_per_dimension + 1
        self.__n_dimensions = len(self.__n_tiles_per_dimension)
        n_tiles_per_tiling = np.prod(self.__n_tiles_per_dimension)
        self.__n_tiles = n_tilings * n_tiles_per_tiling
        self.__tilings = self.__create_tilings(state_space_range)
        self.__n_features = self.__n_tiles * n_actions

    def __create_tilings(
            self,
            state_space_range: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        width = state_space_range[1] - state_space_range[0]
        tile_width = width / self.__n_tiles_per_dimension
        tiling_offset = tile_width / self.__n_tilings

        tilings = np.empty((self.__n_tilings, self.__n_dimensions),
                           dtype=np.ndarray)

        min_value = state_space_range[0]
        max_value = state_space_range[1] + tile_width

        # Create the first tile
        for i in range(self.__n_dimensions):
            tilings[0, i] = np.linspace(
                min_value[i],
                max_value[i],
                num=self.__n_tiles_per_dimension[i] + 1)

        # In order to create the rest tilings,
        # subtract the tiling offset from the previous tiling.
        for i in range(1, self.__n_tilings):
            for j in range(self.__n_dimensions):
                tilings[i, j] = tilings[i - 1, j] - tiling_offset[j]

        return tilings

    def __get_active_features(self, state: np.ndarray) -> np.ndarray:
        active_features = np.zeros((self.__n_tilings,), dtype=np.uint32)
        dimensions = np.append(self.__n_tilings, self.__n_tiles_per_dimension)

        for tiling_i in range(self.__n_tilings):
            index = [tiling_i]
            index += [np.digitize(state[i], self.__tilings[tiling_i, i]) - 1
                      for i in range(self.__n_dimensions)]

            for j in range(len(dimensions)):
                active_features[tiling_i] += (np.prod(dimensions[j + 1:])
                                              * index[j])

        return active_features

    def calculate_q(self,
                    weights: np.ndarray,
                    state: np.ndarray) -> np.ndarray:
        q_values = np.empty((self.__n_actions,))
        active_features = self.__get_active_features(state)
        for action in range(self.__n_actions):
            q_values[action] = np.sum(
                weights[action * self.__n_tiles + active_features])

        return q_values

    def get_features(self, state: np.ndarray, action: int) -> np.ndarray:
        features = np.zeros((self.n_features,))
        active_features = self.__get_active_features(state)
        features[action * self.__n_tiles + active_features] = 1
        return features

    @property
    def n_features(self) -> int:
        return self.__n_features

    def __str__(self) -> str:
        return f'Tile Coding: tilings = {self.__tilings}'
