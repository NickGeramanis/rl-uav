"""This module contains the LSPI class."""
import numpy as np
import rospy
from gym import Env

from rl_uav.features.feature_constructor import FeatureConstructor
from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm


class LSPI(RLAlgorithm):
    """The Least Squares Policy Iteration (LSPI) algorithm."""
    _env: Env
    _discount_factor: float
    _feature_constructor: FeatureConstructor
    _delta: float
    _weights: np.ndarray
    _samples: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 feature_constructor: FeatureConstructor,
                 delta: float) -> None:
        self._env = env
        self._discount_factor = discount_factor
        self._feature_constructor = feature_constructor
        self._delta = delta

        rospy.loginfo(self)

    def gather_samples(self, n_samples: int) -> None:
        """Gather samples by following a random policy."""
        self._samples = np.empty((n_samples,), dtype=tuple)
        samples_gathered = 0
        current_state = self._env.observation_space.sample()
        done = True

        while samples_gathered < n_samples:
            if done:
                current_state = self._env.reset()

            action = self._env.action_space.sample()
            next_state, reward, done, _ = self._env.step(action)
            self._samples[samples_gathered] = (current_state,
                                               action,
                                               reward,
                                               next_state,
                                               done)
            samples_gathered += 1
            current_state = next_state

    def _calculate_features_list(self) -> np.ndarray:
        features_list = np.empty((self._samples.shape[0],), dtype=np.ndarray)

        for i, sample in enumerate(self._samples):
            state = sample[0]
            action = sample[1]
            features_list[i] = self._feature_constructor.get_features(state,
                                                                      action)

        return features_list

    def _lstdq(self, features_list: np.ndarray) -> np.ndarray:
        a_matrix = (self._delta
                    * np.identity(self._feature_constructor.n_features))
        b_matrix = np.zeros((self._feature_constructor.n_features,))

        for i, sample in enumerate(self._samples):
            reward = sample[2]
            next_state = sample[3]
            done = sample[4]

            if done:
                next_features = np.zeros(
                    (self._feature_constructor.n_features,))
            else:
                q_values = self._feature_constructor.calculate_q(
                    self._weights,
                    next_state)
                best_action = np.argmax(q_values)
                next_features = self._feature_constructor.get_features(
                    next_state,
                    best_action)

            current_features = features_list[i]

            a_matrix += np.outer(
                current_features,
                (current_features - self._discount_factor * next_features))
            b_matrix += current_features * reward

        rank = np.linalg.matrix_rank(a_matrix)
        if rank == self._feature_constructor.n_features:
            a_inverse = np.linalg.inv(a_matrix)
        else:
            rospy.logwarn(f'A is not full rank (rank={rank})')
            u_matrix, s_matrix, vh_matrix = np.linalg.svd(a_matrix)
            s_matrix = np.diag(s_matrix)
            a_inverse = np.matmul(np.matmul(vh_matrix.T,
                                            np.linalg.pinv(s_matrix)),
                                  u_matrix.T)

        return np.matmul(a_inverse, b_matrix)

    def train(self, n_episodes: int) -> None:
        new_weights = np.random.random(
            (self._feature_constructor.n_features,))
        features_list = self._calculate_features_list()

        for episode_i in range(n_episodes):
            self._weights = new_weights
            new_weights = self._lstdq(features_list)

            weights_difference = np.linalg.norm(new_weights - self._weights)
            rospy.loginfo(f'episode={episode_i}|'
                          f'weights_difference={weights_difference}')

            if weights_difference <= 0:
                break

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
                action = q_values
                state, reward, done, _ = self._env.step(action)
                episode_reward += reward
                episode_actions += 1

            rospy.loginfo(f'episode={episode_i}|reward={episode_reward}'
                          f'|actions={episode_actions}')

    def __str__(self) -> str:
        return (f'LSPI: discount factor = {self._discount_factor}|'
                f'{self._feature_constructor}')
