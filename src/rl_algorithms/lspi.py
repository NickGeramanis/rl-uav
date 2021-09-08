import logging

import numpy as np

from rl_algorithm import RLAlgorithm


class LSPI(RLAlgorithm):

    def __init__(self, env, discount_factor, feature_constructor):
        RLAlgorithm.__init__(self)
        self.env = env

        self.discount_factor = discount_factor

        self.feature_constructor = feature_constructor
        self.weights = None
        self.sample_set = None

        self._logger.info('LSPI: discount factor = {}'.format(discount_factor))
        self._logger.info(self.feature_constructor.info)

    def gather_samples(self, n_samples):
        self.sample_set = np.empty((n_samples,), dtype=object)
        samples_gathered = 0
        done = True

        while samples_gathered < n_samples:
            if done:
                current_state = self.env.reset()

            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.sample_set[samples_gathered] = (
                current_state, action, reward, next_state, done)
            samples_gathered += 1
            current_state = next_state

    def calculate_features_list(self):
        features_list = np.empty((self.sample_set.shape[0],), dtype=object)
        sample_i = 0

        for sample in self.sample_set:
            current_state = sample[0]
            action = sample[1]
            features_list[sample_i] = self.feature_constructor.get_features(
                current_state, action)
            sample_i += 1

        return features_list

    def lstdq(self, features_list, delta):
        a = delta * np.identity(self.feature_constructor.n_features)
        b = np.zeros((self.feature_constructor.n_features,))
        sample_i = 0

        for sample in self.sample_set:
            current_state = sample[0]
            action = sample[1]
            reward = sample[2]
            next_state = sample[3]
            done = sample[4]
            if done:
                next_features = np.zeros(
                    (self.feature_constructor.n_features,))
            else:
                best_action = np.argmax(
                    self.feature_constructor.calculate_q(
                        self.weights, next_state))
                next_features = self.feature_constructor.get_features(
                    next_state, best_action)

            if features_list is not None:
                current_features = features_list[sample_i]
                sample_i += 1
            else:
                current_features = self.feature_constructor.get_features(
                    current_state, action)

            a += np.outer(
                current_features,
                (current_features - self.discount_factor * next_features))
            b += current_features * reward

        rank = np.linalg.matrix_rank(a)
        if rank == self.feature_constructor.n_features:
            a_inverse = np.linalg.inv(a)
        else:
            self._logger.warning("A is not full rank (rank={})".format(rank))
            u, s, vh = np.linalg.svd(a)
            s = np.diag(s)
            a_inverse = np.matmul(np.matmul(vh.T, np.linalg.pinv(s)), u.T)

        return np.matmul(a_inverse, b)

    def train(self, training_episodes, tolerance=0, delta=0,
              pre_calculate_features=False):
        new_weights = np.random.random((self.feature_constructor.n_features,))
        if pre_calculate_features:
            features_list = self.calculate_features_list()
        else:
            features_list = None

        for episode_i in range(training_episodes):
            self.weights = new_weights
            new_weights = self.lstdq(features_list, delta)

            weights_difference = np.linalg.norm(new_weights - self.weights)
            self._logger.info('episode={}|weights_difference={}'.format(
                episode_i, weights_difference))

            if weights_difference <= tolerance:
                break

    def run(self, episodes, render=False):
        for episode_i in range(episodes):
            episode_reward = 0.0
            episode_actions = 0
            state = self.env.reset()
            done = False

            while not done:
                if render:
                    self.env.render()

                action = np.argmax(
                    self.feature_constructor.calculate_q(self.weights, state))
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_actions += 1

            self._logger.info('episode={}|reward={}|actions={}'.format(
                episode_i, episode_reward, episode_actions))
