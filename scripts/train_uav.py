#!/usr/bin/env python
import numpy as np
import rospy

from envs.navigation_env import NavigationEnv
from features.rbf import RBF
from features.tile_coding import TileCoding
from rl_algorithms.lfa_sarsa_lambda import LFASARSALambda
from rl_algorithms.lspi import LSPI


def main():
    rospy.init_node('train_uav')
    env = NavigationEnv()

    training_episodes = 500
    run_episodes = 3
    weights_filename = 'weights.npy'
    samples_filename = 'navigation_samples.npy'

    discount_factor = 0.99
    learning_rate_steepness = 0.02
    learning_rate_midpoint = 350
    lambda_ = 0.5

    # Feature Constructor
    tiles_per_dimension = [14, 14, 14, 14, 14]
    displacement_vector = [1, 1, 1, 1, 1]
    n_tilings = 7
    initial_learning_rate = 0.1 / n_tilings
    feature_constructor = TileCoding(
        env.action_space.n, n_tilings, tiles_per_dimension,
        env.observation_space, displacement_vector)

    # SARSA(lambda) with Linear Function Approximation
    lfa_sarsa_lambda = LFASARSALambda(
        env, learning_rate_midpoint,
        discount_factor, initial_learning_rate, learning_rate_steepness,
        feature_constructor, lambda_)
    lfa_sarsa_lambda.train(training_episodes)
    np.save(weights_filename, lfa_sarsa_lambda.weights)
    np.save(samples_filename, lfa_sarsa_lambda.sample_set, allow_pickle=True)
    lfa_sarsa_lambda.weights = np.load(weights_filename)
    lfa_sarsa_lambda.run(run_episodes)

    # Least-Squares Policy Iteration
    tolerance = 0
    delta = 0.1
    pre_calculate_features = True
    lspi = LSPI(env, discount_factor, feature_constructor)
    lspi.sample_set = np.load(samples_filename, allow_pickle=True)
    lspi.train(training_episodes, tolerance, delta, pre_calculate_features)
    np.save(weights_filename, lspi.weights)
    lspi.weights = np.load(weights_filename)
    lspi.run(run_episodes)


if __name__ == '__main__':
    main()
