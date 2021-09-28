import math
import random

import numpy as np
import rospy

from rl_uav.envs.env import Env
from rl_uav.features.feature_constructor import FeatureConstructor
from rl_uav.rl_algorithms.rl_algorithm import RLAlgorithm


class LFASARSALambda(RLAlgorithm):
    __env: Env
    __discount_factor: float
    __initial_learning_rate: float
    __learning_rate_midpoint: int
    __learning_rate_steepness: float
    __feature_constructor: FeatureConstructor
    __lambda: float
    __weights: np.ndarray

    def __init__(self,
                 env: Env,
                 discount_factor: float,
                 initial_learning_rate: float,
                 learning_rate_midpoint: int,
                 learning_rate_steepness: float,
                 feature_constructor: FeatureConstructor,
                 lambda_: float) -> None:
        self.__env = env
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__learning_rate_steepness = learning_rate_steepness
        self.__feature_constructor = feature_constructor
        self.__lambda = lambda_
        self.__weights = np.random.random((feature_constructor.n_features,))

        rospy.loginfo(self)

    def train(self, n_episodes: int) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0

            try:
                exponent = (self.__learning_rate_steepness
                            * (episode_i - self.__learning_rate_midpoint))
                learning_rate = (self.__initial_learning_rate
                                 / (1 + math.exp(exponent)))
            except OverflowError:
                learning_rate = 0

            try:
                epsilon = 1.0 / (episode_i + 1)
            except OverflowError:
                epsilon = 0

            done = False
            current_state = self.__env.reset()
            eligibility_traces = np.zeros(
                (self.__feature_constructor.n_features,))
            current_q = self.__feature_constructor.calculate_q(self.__weights,
                                                               current_state)

            if random.random() <= epsilon:
                current_action = self.__env.action_space.sample()
            else:
                current_action = np.argmax(current_q)

            while not done:
                next_state, reward, done, _ = self.__env.step(current_action)
                episode_reward += reward
                episode_actions += 1

                next_q = self.__feature_constructor.calculate_q(self.__weights,
                                                                next_state)

                if random.random() <= epsilon:
                    next_action = self.__env.action_space.sample()
                else:
                    next_action = np.argmax(next_q)

                td_target = reward
                if not done:
                    td_target += self.__discount_factor * next_q[next_action]

                td_error = td_target - current_q[current_action]

                current_features = self.__feature_constructor.get_features(
                    current_state,
                    current_action)
                eligibility_traces = (self.__discount_factor
                                      * self.__lambda
                                      * eligibility_traces
                                      + current_features)

                self.__weights += (learning_rate
                                   * td_error
                                   * eligibility_traces)

                current_state = next_state
                current_action = next_action
                current_q = next_q

            rospy.loginfo(f'episode={episode_i}|reward={episode_reward}'
                          f'|actions={episode_actions}')

    def run(self, n_episodes: int, render: bool = False) -> None:
        for episode_i in range(n_episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.__env.reset()
            done = False

            while not done:
                if render:
                    self.__env.render()

                q = self.__feature_constructor.calculate_q(self.__weights,
                                                           state)
                action = np.argmax(q)
                state, reward, done, _ = self.__env.step(action)
                episode_reward += reward
                episode_actions += 1

            rospy.loginfo(f'episode={episode_i}|reward={episode_reward}'
                          f'|actions={episode_actions}')

    def __str__(self) -> str:
        return ('SARSA(lambda) with Linear Function Approximation:'
                f'discount factor={self.__discount_factor}|'
                f'initial learning rate = {self.__initial_learning_rate}|'
                f'learning rate midpoint = {self.__learning_rate_midpoint}|'
                f'learning rate steepness = {self.__learning_rate_steepness}|'
                f'lambda = {self.__lambda}|'
                f'{self.__feature_constructor}')
