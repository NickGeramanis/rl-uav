"""This module contains the LFASARSALambda class."""
import math
import random

import numpy as np
import rospy
from gym import Env

from rl_uav.features.feature_constructor import FeatureConstructor
from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm


class LFASARSALambda(RLAlgorithm):
    """The SARSA(lambda) algorithm with Linear Function Approximation."""
    _env: Env
    _discount_factor: float
    _initial_learning_rate: float
    _learning_rate_steepness: float
    _feature_constructor: FeatureConstructor
    _lambda: float
    _weights: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_steepness: float,
                 feature_constructor: FeatureConstructor,
                 lambda_: float) -> None:
        self._env = env
        self._discount_factor = discount_factor
        self._initial_learning_rate = initial_learning_rate
        self._learning_rate_steepness = learning_rate_steepness
        self._feature_constructor = feature_constructor
        self._lambda = lambda_
        self._weights = np.random.random((feature_constructor.n_features,))

        rospy.loginfo(self)

    def train(self, n_episodes: int) -> None:
        learning_rate_midpoint = n_episodes / 2
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                exponent = (self._learning_rate_steepness
                            * (episode_i - learning_rate_midpoint))
                learning_rate = (self._initial_learning_rate
                                 / (1 + math.exp(exponent)))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            current_state = self._env.reset()
            eligibility_traces = np.zeros(
                (self._feature_constructor.n_features,))
            current_q_values = self._feature_constructor.calculate_q(
                self._weights,
                np.array(current_state))

            if random.random() <= epsilon:
                current_action = self._env.action_space.sample()
            else:
                current_action = np.argmax(current_q_values)

            while not done:
                next_state, reward, done, _ = self._env.step(current_action)
                episode_reward += reward
                episode_actions += 1

                next_q_values = self._feature_constructor.calculate_q(
                    self._weights,
                    np.array(next_state))

                if random.random() <= epsilon:
                    next_action = self._env.action_space.sample()
                else:
                    next_action = np.argmax(next_q_values)

                td_target = reward
                if not done:
                    td_target += (self._discount_factor
                                  * next_q_values[next_action])

                td_error = td_target - current_q_values[current_action]

                current_features = self._feature_constructor.get_features(
                    np.array(current_state),
                    current_action)
                eligibility_traces = (self._discount_factor
                                      * self._lambda
                                      * eligibility_traces
                                      + current_features)

                self._weights += (learning_rate
                                  * td_error
                                  * eligibility_traces)

                current_state = next_state
                current_action = next_action
                current_q_values = next_q_values

            rospy.loginfo(f'episode={episode_i}|reward={episode_reward}'
                          f'|actions={episode_actions}')

    def run(self, n_episodes: int, render: bool = False) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self._env.reset()
            done = False

            while not done:
                if render:
                    self._env.render()

                q_values = self._feature_constructor.calculate_q(
                    self._weights,
                    np.array(state))
                action = np.argmax(q_values)
                state, reward, done, _ = self._env.step(action)
                episode_reward += reward
                episode_actions += 1

            rospy.loginfo(f'episode={episode_i}|reward={episode_reward}'
                          f'|actions={episode_actions}')

    def __str__(self) -> str:
        return ('SARSA(lambda) with Linear Function Approximation:'
                f'discount factor={self._discount_factor}|'
                f'initial learning rate = {self._initial_learning_rate}|'
                f'learning rate steepness = {self._learning_rate_steepness}|'
                f'lambda = {self._lambda}|'
                f'{self._feature_constructor}')
