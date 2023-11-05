"""This module contains the TabularMonteCarlo class."""
import random

import numpy as np
from gymnasium import Env

from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm
from rl_uav.features.discretizer import Discretizer


class TabularMonteCarlo(RLAlgorithm):
    """The Tabular Monte Carlo algorithm."""
    _env: Env
    _discount_factor: float
    _discretizer: Discretizer
    _q_table: np.ndarray
    _returns: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 discretizer: Discretizer) -> None:
        super().__init__()
        self._env = env
        self._discount_factor = discount_factor
        self._discretizer = discretizer
        self._q_table = np.random.random(
            (discretizer.n_bins + (self._env.action_space.n,)))
        self._returns = np.empty(self._q_table.shape, dtype=list)

    def train(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0.1

            samples = []
            terminated = truncated = False
            state, _ = self._env.reset()
            state = self._discretizer.discretize(state)
            while not terminated or not truncated:
                if random.random() <= epsilon:
                    action = self._env.action_space.sample()
                else:
                    action = np.argmax(self._q_table[state])

                state, reward, terminated, truncated, _ = self._env.step(action)
                state = self._discretizer.discretize(state)
                episode_reward += reward

                samples.append((state, action, reward))

            self._logger.info('episode=%d|reward=%f',
                              episode_i,
                              episode_reward)

            return_ = 0.0
            processed_samples = []
            for sample in reversed(samples):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                return_ = self._discount_factor * return_ + reward

                if (state, action) in processed_samples:
                    continue

                processed_samples.append((state, action))
                if self._returns[state + (action,)] is None:
                    self._returns[state + (action,)] = [return_]
                else:
                    self._returns[state + (action,)].append(return_)

                self._q_table[state + (action,)] = (
                    np.mean(self._returns[state + (action,)]))

    def run(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            state, _ = self._env.reset()
            terminated = truncated = False

            while not terminated or not truncated:
                state = self._discretizer.discretize(state)
                action = np.argmax(self._q_table[state])
                state, reward, terminated, truncated, _ = self._env.step(action)
                episode_reward += reward

            self._logger.info('episode=%d|reward=%f',
                              episode_i,
                              episode_reward)
