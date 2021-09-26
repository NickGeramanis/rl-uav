import math
import random

import numpy as np

from src.rl_algorithms.rl_algorithm import RLAlgorithm


class LFASARSALambda(RLAlgorithm):

    def __init__(self,
                 env,
                 discount_factor,
                 initial_learning_rate,
                 learning_rate_midpoint,
                 learning_rate_steepness,
                 feature_constructor,
                 lambda_):
        super(LFASARSALambda, self).__init__('info.log')
        self.__env = env
        self.__discount_factor = discount_factor
        self.__initial_learning_rate = initial_learning_rate
        self.__learning_rate_midpoint = learning_rate_midpoint
        self.__learning_rate_steepness = learning_rate_steepness
        self.__feature_constructor = feature_constructor
        self.__lambda = lambda_
        self.__weights = np.random.random((feature_constructor.n_features,))

        self._logger.info(self)

    def train(self, n_episodes):
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

            self._logger.info('episode=%d|reward=%f|actions=%d',
                              episode_i, episode_reward, episode_actions)

    def run(self, n_episodes, render=False):
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

            self._logger.info('episode=%d|reward=%f|actions=%d',
                              episode_i, episode_reward, episode_actions)

    def __str__(self):
        return ('SARSA(lambda) with Linear Function Approximation:'
                'discount factor={}|initial learning rate = {}|'
                'learning rate midpoint = {}|learning rate steepness = {}|'
                'lambda = {}|{}').format(self.__discount_factor,
                                         self.__initial_learning_rate,
                                         self.__learning_rate_midpoint,
                                         self.__learning_rate_steepness,
                                         self.__lambda,
                                         self.__feature_constructor)
