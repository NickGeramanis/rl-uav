"""This module contains the TabularQLambda class."""
import math
import random

import numpy as np
from gymnasium import Env

from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm
from rl_uav.features.discretizer import Discretizer


class TabularQLambda(RLAlgorithm):
    """The Tabular Q(lambda) algorithm."""
    _env: Env
    _discount_factor: float
    _initial_learning_rate: float
    _learning_rate_steepness: float
    _discretizer: Discretizer
    _lambda: float
    _q_table: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_steepness: float,
                 discretizer: Discretizer,
                 lambda_: float) -> None:
        super().__init__()
        self._env = env
        self._discount_factor = discount_factor
        self._initial_learning_rate = initial_learning_rate
        self._learning_rate_steepness = learning_rate_steepness
        self._discretizer = discretizer
        self._lambda = lambda_
        self._q_table = np.random.random(
            (discretizer.n_bins + (self._env.action_space.n,)))

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
            eligibility_traces = np.zeros(
                (self._discretizer.n_bins + (self._env.action_space.n,)))
            current_state, _ = self._env.reset()
            current_state = self._discretizer.discretize(current_state)

            if random.random() <= epsilon:
                current_action = self._env.action_space.sample()
            else:
                current_action = np.argmax(self._q_table[current_state])

            while not terminated and not truncated:
                next_state, reward, terminated, truncated, _ = self._env.step(current_action)
                next_state = self._discretizer.discretize(next_state)
                episode_reward += reward

                if random.random() <= epsilon:
                    next_action = self._env.action_space.sample()
                else:
                    next_action = np.argmax(self._q_table[next_state])

                if (self._q_table[next_state + (next_action,)]
                        == np.max(self._q_table[next_state])):
                    best_action = next_action
                else:
                    best_action = np.argmax(self._q_table[next_state])

                td_target = reward
                if not terminated:
                    td_target += (
                            self._discount_factor
                            * self._q_table[next_state + (next_action,)])

                td_error = (
                        td_target
                        - self._q_table[current_state + (current_action,)])
                eligibility_traces[current_state + (current_action,)] += 1

                self._q_table += learning_rate * td_error * eligibility_traces

                if best_action == next_action:
                    eligibility_traces *= (self._discount_factor
                                           * self._lambda)
                else:
                    eligibility_traces *= 0

                current_state = next_state
                current_action = next_action

            self._logger.info('episode=%d|reward=%f',
                              episode_i,
                              episode_reward)

    def run(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            state, _ = self._env.reset()
            terminated = truncated = False

            while not terminated and not truncated:
                state = self._discretizer.discretize(state)
                action = np.argmax(self._q_table[state])
                state, reward, terminated, truncated, _ = self._env.step(action)
                episode_reward += reward

            self._logger.info('episode=%d|reward=%f',
                              episode_i,
                              episode_reward)
