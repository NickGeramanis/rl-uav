#!/usr/bin/env python3
import numpy as np
import rospy

from rl_uav.envs.navigation import Navigation
from rl_uav.features.radial_basis_functions import RadialBasisFunctions
from rl_uav.features.tile_coding import TileCoding
from rl_uav.rl_algorithms.lfa_sarsa_lambda import LFASARSALambda
from rl_uav.rl_algorithms.lspi import LSPI


def main():
    rospy.init_node('train_uav')
    env = Navigation(track_id=1)

    n_episodes = 500

    discount_factor = 0.99
    state_space_range = (env.observation_space.low, env.observation_space.high)
    learning_rate_steepness = 0.02
    lambda_ = 0.5

    # Tile Coding
    n_tiles_per_dimension = np.array([14, 14, 14, 14, 14])
    n_tilings = 7
    initial_learning_rate = 0.1 / n_tilings
    feature_constructor1 = TileCoding(env.action_space.n,
                                      n_tilings,
                                      n_tiles_per_dimension,
                                      state_space_range)

    # Radial Basis Functions
    standard_deviation = 0.25
    centers_per_dimension = [
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8],
        [0.2, 0.4, 0.6, 0.8]
    ]
    feature_constructor2 = RadialBasisFunctions(env.action_space.n,
                                                state_space_range,
                                                centers_per_dimension,
                                                standard_deviation)

    # SARSA(lambda) with Linear Function Approximation
    lfa_sarsa_lambda = LFASARSALambda(env,
                                      discount_factor,
                                      initial_learning_rate,
                                      learning_rate_steepness,
                                      feature_constructor1,
                                      lambda_)
    lfa_sarsa_lambda.train(n_episodes)

    # Least-Squares Policy Iteration
    n_samples = 1000
    lspi = LSPI(env, discount_factor, feature_constructor2)
    lspi.gather_samples(n_samples)
    lspi.train(n_episodes)
    lspi.run(n_episodes)


if __name__ == '__main__':
    main()
