from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Product

from advertising.regressors.DiscreteRegressor import DiscreteRegressor


class DiscreteGPRegressor(DiscreteRegressor):
    """
    1D-input Gaussian Process Regressor in order to estimate a function
    """

    def __init__(self, arms, init_std_dev=1e3, alpha: float = 10, n_restarts_optimizer: int = 5,
                 normalized: bool = True):
        if normalized:
            arms = arms / np.max(arms)
        super().__init__(arms, init_std_dev)
        self.collected_rewards: List[float] = []

        kernel: Product = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                                                     n_restarts_optimizer=n_restarts_optimizer)

    def update_model(self, pulled_arm: int, reward: float):
        self.pulled_arm_list.append(pulled_arm)
        self.collected_rewards.append(reward)
        self.rewards_per_arm[pulled_arm].append(reward)

        x = np.atleast_2d(np.array(self.arms)[self.pulled_arm_list]).T
        self.gp.fit(x, self.collected_rewards)

    def sample_distribution(self):
        means, sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        sigmas = np.maximum(sigmas, 1e-2)  # avoid negative numbers
        return np.random.normal(means, sigmas)
