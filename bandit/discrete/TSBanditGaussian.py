from typing import List

import numpy as np

from advertising.regressors.DiscreteGaussianRegressor import DiscreteGaussianRegressor
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.discrete.DiscreteBandit import DiscreteBandit


class TSBanditGP(DiscreteBandit):
    """
    Gaussian Process Thompson Sampling Bandit for gaussian distributions
    """

    def __init__(self, arms: List[float], init_std_dev: float = 1e3):
        """
        :param arms: list of non-normalized arms that can be pulled
        :param init_std_dev: initial value of the standard deviation for the prior
        """
        super().__init__(len(arms))

        self.arms: List[float] = arms
        self.regressor: DiscreteRegressor = DiscreteGaussianRegressor(arms, init_std_dev)

    def update(self, pulled_arm: int, reward: float):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.regressor.update_model(pulled_arm, reward)

    def pull_arm(self):
        prob_per_arm = self.regressor.sample_distribution()
        return np.argmax(prob_per_arm)
