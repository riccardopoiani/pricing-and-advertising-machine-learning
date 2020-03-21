import numpy as np

from typing import List

from bandit.discrete.DiscreteBandit import DiscreteBandit


class LinPHEBernoulli(DiscreteBandit):
    """
    "Pertubated-history exploration in stochastic linear bandits" - Kveton et. al [2019]
    """

    def __init__(self, n_arms: int, perturbation: int, regularization: float,
                 features: np.array, features_dim: int):
        super().__init__(n_arms=n_arms)

        # Hyper-parameters
        self.perturbation = perturbation
        self.regularization = regularization

        # Features
        self.features: np.array = features
        self.feature_dim = features_dim

        # Additional data structure
        self.arm_count: np.array = np.zeros(shape=n_arms)
        self.cum_reward_per_arm = np.zeros(shape=n_arms)
        self.pulled_arm_history: List = []

        # G matrix
        self.g: np.array = self.regularization * (1 + self.perturbation) * np.eye(features_dim)
        self.g_inv: np.array = np.linalg.inv(self.g)

        # Estimated parameters
        self.theta_hat = np.zeros(shape=self.feature_dim)

    def _update_g(self):
        """
        Update the matrix G in an efficient way

        :return: none
        """
        curr_features = self.features[self.pulled_arm_history[self.t]]
        self.g += (self.perturbation + 1) * np.outer(curr_features, curr_features)

    def _update_g_inv(self):
        """
        Update the matrix G using the Sharman-Morrison formula # TODO

        :return: none
        """
        self.g_inv = np.linalg.inv(self.g)

    def _update_theta_hat(self):
        """
        Update parameter estimation

        :return: none
        """
        cum_vect = 0.0
        for arm in range(self.n_arms):
            curr_feature = self.features[arm]
            curr_n_binomial = self.perturbation * self.arm_count[arm]
            cum_vect += curr_feature * (self.cum_reward_per_arm[arm] + np.random.binomial(n=curr_n_binomial, p=1 / 2))

        self.theta_hat = np.dot(self.g_inv, cum_vect)

    def pull_arm(self) -> int:
        """
        Retrive the arm to be pulled, according to the current best estimated parameter.
        Exploration is guaranteed due to history perturbation

        :return: index of the arm to be pulled
        """
        if self.t >= self.n_arms:
            estimated_arm_values = np.dot(self.features, self.theta_hat)
            max_value = estimated_arm_values.max()
            idxes = np.argwhere(estimated_arm_values == max_value).reshape(-1)
            idx = np.random.choice(idxes)
        else:
            idx = self.t

        return idx

    def update(self, pulled_arm, reward):
        """
        Update the bandit statistics and the bandit model (i.e. the G matrix, its inverse, and the estimated
        parameters) according to the new observed information

        :param pulled_arm: index of the arm that has been pulled
        :param reward: observed reward
        :return: none
        """
        # Update statistics
        self.arm_count[pulled_arm] += 1
        self.cum_reward_per_arm[pulled_arm] += reward
        self.pulled_arm_history.append(pulled_arm)

        self.update_observations(pulled_arm=pulled_arm, reward=reward)

        # Update model information
        self._update_g()
        self._update_g_inv()
        self._update_theta_hat()

        # Update time
        self.t += 1
