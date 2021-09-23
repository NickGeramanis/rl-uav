import numpy as np

from src.rl_algorithms.rl_algorithm import RLAlgorithm


class LSPI(RLAlgorithm):
    __env = None
    __discount_factor = None
    __feature_constructor = None
    __tolerance = None
    __delta = None
    __weights = None
    __samples = None

    def __init__(self, env, discount_factor, feature_constructor, tolerance,
                 delta):
        super(LSPI, self).__init__('info.log')
        self.__env = env
        self.__discount_factor = discount_factor
        self.__feature_constructor = feature_constructor
        self.__tolerance = tolerance
        self.__delta = delta

        self._logger.info(self)

    def gather_samples(self, n_samples):
        self.__samples = np.empty((n_samples,), dtype=tuple)
        samples_gathered = 0
        current_state = self.__env.observation_space.sample()
        done = True

        while samples_gathered < n_samples:
            if done:
                current_state = self.__env.reset()

            action = self.__env.action_space.sample()
            next_state, reward, done, _ = self.__env.step(action)
            self.__samples[samples_gathered] = (current_state, action, reward,
                                                next_state, done)
            samples_gathered += 1
            current_state = next_state

    def __calculate_features_list(self):
        features_list = np.empty((self.__samples.shape[0],), dtype=np.ndarray)

        for i, sample in enumerate(self.__samples):
            state = sample[0]
            action = sample[1]
            features_list[i] = self.__feature_constructor.get_features(state,
                                                                       action)

        return features_list

    def __lstdq(self, features_list):
        a = self.__delta * np.identity(self.__feature_constructor.n_features)
        b = np.zeros((self.__feature_constructor.n_features,))

        for i, sample in enumerate(self.__samples):
            reward = sample[2]
            next_state = sample[3]
            done = sample[4]

            if done:
                next_features = np.zeros(
                    (self.__feature_constructor.n_features,))
            else:
                q = self.__feature_constructor.calculate_q(self.__weights,
                                                           next_state)
                best_action = int(np.argmax(q))
                next_features = self.__feature_constructor.get_features(
                    next_state, best_action)

            current_features = features_list[i]

            a += np.outer(
                current_features,
                (current_features - self.__discount_factor * next_features))
            b += current_features * reward

        rank = np.linalg.matrix_rank(a)
        if rank == self.__feature_constructor.n_features:
            a_inverse = np.linalg.inv(a)
        else:
            self._logger.warning('A is not full rank (rank=%d)', rank)
            u, s, vh = np.linalg.svd(a)
            s = np.diag(s)
            a_inverse = np.matmul(np.matmul(vh.T, np.linalg.pinv(s)), u.T)

        return np.matmul(a_inverse, b)

    def train(self, n_episodes):
        new_weights = np.random.random(
            (self.__feature_constructor.n_features,))
        features_list = self.__calculate_features_list()

        for episode_i in range(n_episodes):
            self.__weights = new_weights
            new_weights = self.__lstdq(features_list)

            weights_difference = np.linalg.norm(new_weights - self.__weights)
            self._logger.info('episode=%d|weights_difference=%f',
                              episode_i, weights_difference)

            if weights_difference <= self.__tolerance:
                break

    def run(self, n_episodes, renderFalse):
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
        return 'LSPI: discount factor={}|{}'.format(self.__discount_factor,
                                                    self.__feature_constructor)
