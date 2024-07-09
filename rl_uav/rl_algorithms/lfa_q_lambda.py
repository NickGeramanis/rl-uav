"""This module contains the LFAQLambda class."""
import math
import random

import numpy as np
from gymnasium import Env

from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm
from rl_uav.features.linear_function_approximation import LinearFunctionApproximation


class LFAQLambda(RLAlgorithm):
    """The Q(lambda) algorithm with Linear Function Approximation."""
    _env: Env
    _discount_factor: float
    _initial_learning_rate: float
    _learning_rate_steepness: float
    _feature_constructor: LinearFunctionApproximation
    _lambda: float
    _weights: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_steepness: float,
                 feature_constructor: LinearFunctionApproximation,
                 lambda_: float) -> None:
        super().__init__()
        self._env = env
        self._discount_factor = discount_factor
        self._initial_learning_rate = initial_learning_rate
        self._learning_rate_steepness = learning_rate_steepness
        self._feature_constructor = feature_constructor
        self._lambda = lambda_
        self._weights = np.random.random((feature_constructor.n_features,))

    def train(self, n_episodes: int) -> None:
        learning_rate_midpoint = n_episodes / 2
        for episode_i in range(n_episodes):
            episode_reward = 0.0

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

            terminated = truncated = False
            current_state, _ = self._env.reset()
            eligibility_traces = np.zeros(
                (self._feature_constructor.n_features,))
            current_q_values = self._feature_constructor.calculate_q(
                self._weights,
                current_state)

            if random.random() <= epsilon:
                current_action = self._env.action_space.sample()
            else:
                current_action = np.argmax(current_q_values)

            while not terminated and not truncated:
                next_state, reward, terminated, truncated, _ = self._env.step(
                    current_action)
                episode_reward += reward

                next_q_values = self._feature_constructor.calculate_q(
                    self._weights,
                    next_state)

                if random.random() <= epsilon:
                    next_action = self._env.action_space.sample()
                else:
                    next_action = np.argmax(next_q_values)

                if next_q_values[next_action] == np.max(next_q_values):
                    best_action = next_action
                else:
                    best_action = np.argmax(next_q_values)

                td_target = reward
                if not terminated:
                    td_target += (self._discount_factor
                                  * next_q_values[next_action])

                td_error = td_target - current_q_values[current_action]

                if best_action == next_action:
                    features = self._feature_constructor.get_features(
                        current_state,
                        current_action)
                    eligibility_traces = (self._discount_factor
                                          * self._lambda
                                          * eligibility_traces
                                          + features)
                else:
                    eligibility_traces *= 0

                self._weights += learning_rate * td_error * eligibility_traces

                current_state = next_state
                current_action = next_action
                current_q_values = next_q_values

            self._logger.info('episode=%d|reward=%f',
                              episode_i,
                              episode_reward)

    def run(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            state, _ = self._env.reset()
            terminated = truncated = False

            while not terminated and not truncated:
                q_values = self._feature_constructor.calculate_q(self._weights,
                                                                 state)
                action = np.argmax(q_values)
                state, reward, terminated, truncated, _ = self._env.step(
                    action)
                episode_reward += reward

            self._logger.info('episode=%d|reward=%f',
                              episode_i,
                              episode_reward)
