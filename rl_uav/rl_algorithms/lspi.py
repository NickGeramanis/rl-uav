"""This module contains the LSPI class."""
import numpy as np
import rospy
from gym import Env

from rl_uav.features.feature_constructor import FeatureConstructor
from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm


class LSPI(RLAlgorithm):
    """The Least Squares Policy Iteration (LSPI) algorithm."""
    __env: Env
    __discount_factor: float
    __feature_constructor: FeatureConstructor
    __delta: float
    __weights: np.ndarray
    __samples: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 feature_constructor: FeatureConstructor,
                 delta: float) -> None:
        self.__env = env
        self.__discount_factor = discount_factor
        self.__feature_constructor = feature_constructor
        self.__delta = delta

        rospy.loginfo(self)

    def gather_samples(self, n_samples: int) -> None:
        """Gather samples by following a random policy."""
        self.__samples = np.empty((n_samples,), dtype=tuple)
        samples_gathered = 0
        current_state = self.__env.observation_space.sample()
        done = True

        while samples_gathered < n_samples:
            if done:
                current_state = self.__env.reset()

            action = self.__env.action_space.sample()
            next_state, reward, done, _ = self.__env.step(action)
            self.__samples[samples_gathered] = (current_state,
                                                action,
                                                reward,
                                                next_state,
                                                done)
            samples_gathered += 1
            current_state = next_state

    def __calculate_features_list(self) -> np.ndarray:
        features_list = np.empty((self.__samples.shape[0],), dtype=np.ndarray)

        for i, sample in enumerate(self.__samples):
            state = sample[0]
            action = sample[1]
            features_list[i] = self.__feature_constructor.get_features(state,
                                                                       action)

        return features_list

    def __lstdq(self, features_list: np.ndarray) -> np.ndarray:
        a_matrix = (self.__delta
                    * np.identity(self.__feature_constructor.n_features))
        b_matrix = np.zeros((self.__feature_constructor.n_features,))

        for i, sample in enumerate(self.__samples):
            reward = sample[2]
            next_state = sample[3]
            done = sample[4]

            if done:
                next_features = np.zeros(
                    (self.__feature_constructor.n_features,))
            else:
                q_values = self.__feature_constructor.calculate_q(
                    self.__weights,
                    next_state)
                best_action = np.argmax(q_values)
                next_features = self.__feature_constructor.get_features(
                    next_state,
                    best_action)

            current_features = features_list[i]

            a_matrix += np.outer(
                current_features,
                (current_features - self.__discount_factor * next_features))
            b_matrix += current_features * reward

        rank = np.linalg.matrix_rank(a_matrix)
        if rank == self.__feature_constructor.n_features:
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
            (self.__feature_constructor.n_features,))
        features_list = self.__calculate_features_list()

        for episode_i in range(n_episodes):
            self.__weights = new_weights
            new_weights = self.__lstdq(features_list)

            weights_difference = np.linalg.norm(new_weights - self.__weights)
            rospy.loginfo(f'episode={episode_i}|'
                          f'weights_difference={weights_difference}')

            if weights_difference <= 0:
                break

    def run(self, n_episodes: int, render: bool = False) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.__env.reset()
            done = False

            while not done:
                if render:
                    self.__env.render()

                q_values = self.__feature_constructor.calculate_q(
                    self.__weights,
                    np.array(state))
                action = q_values
                state, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1

            rospy.loginfo(f'episode={episode_i}|reward={episode_reward}'
                          f'|actions={episode_actions}')

    def __str__(self) -> str:
        return (f'LSPI: discount factor = {self.__discount_factor}|'
                f'{self.__feature_constructor}')
