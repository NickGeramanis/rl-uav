#!/usr/bin/env python3
import numpy as np
import rospy

from rl_uav.envs.navigation_env import NavigationEnv
from rl_uav.features.radial_basis_functions import RadialBasisFunctions
from rl_uav.features.tile_coding import TileCoding
from rl_uav.rl_algorithms.lfa_sarsa_lambda import LFASARSALambda
from rl_uav.rl_algorithms.lspi import LSPI


def main():
    rospy.init_node('train_uav')
    env = NavigationEnv(1)

    n_episodes = 500

    discount_factor = 0.99
    state_space_low = env.observation_space.low
    state_space_high = env.observation_space.high
    learning_rate_steepness = 0.02
    learning_rate_midpoint = 350
    lambda_ = 0.5

    # Tile Coding
    n_tiles_per_dimension = np.array([14, 14, 14, 14, 14])
    displacement_vector = np.array([1, 1, 1, 1, 1])
    n_tilings = 7
    initial_learning_rate = 0.1 / n_tilings
    feature_constructor1 = TileCoding(env.action_space.n,
                                      n_tilings,
                                      n_tiles_per_dimension,
                                      state_space_low,
                                      state_space_high,
                                      displacement_vector)

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
                                                state_space_low,
                                                state_space_high,
                                                centers_per_dimension,
                                                standard_deviation)

    # SARSA(lambda) with Linear Function Approximation
    lfa_sarsa_lambda = LFASARSALambda(env,
                                      discount_factor,
                                      initial_learning_rate,
                                      learning_rate_midpoint,
                                      learning_rate_steepness,
                                      feature_constructor1,
                                      lambda_)
    lfa_sarsa_lambda.train(n_episodes)

    # Least-Squares Policy Iteration
    tolerance = 0
    delta = 0.1
    n_samples = 1000
    lspi = LSPI(env, discount_factor, feature_constructor2, tolerance, delta)
    lspi.gather_samples(n_samples)
    lspi.train(n_episodes)
    lspi.run(n_episodes)


if __name__ == '__main__':
    main()
