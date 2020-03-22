from typing import List
import numpy as np

from advertising.regressors.GP_Regressor import GP_Regressor
from bandit.discrete.DiscreteBandit import DiscreteBandit


class GP_TSBandit(DiscreteBandit):
    """
    Gaussian Process Thompson Sampling Bandit for gaussian distributions
    """

    def __init__(self, arms: List[float], alpha: float = 10, n_restarts_optimizer: int = 5):
        super().__init__(len(arms))

        self.arms: List[float] = arms
        self.gp_regressor: GP_Regressor = GP_Regressor(alpha, n_restarts_optimizer)

    def update(self, pulled_arm: int, reward: float):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.gp_regressor.update_model(self.arms[pulled_arm], reward)

    def pull_arm(self):
        prob_per_arm = self.gp_regressor.sample_gp_distribution(self.arms)
        return np.argmax(prob_per_arm)



