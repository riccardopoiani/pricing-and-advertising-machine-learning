from typing import List
import numpy as np

from advertising.regressors.DiscreteGPRegressor import DiscreteGPRegressor
from bandit.discrete.DiscreteBandit import DiscreteBandit


class TSBanditGP(DiscreteBandit):
    """
    Gaussian Process Thompson Sampling Bandit for gaussian distributions
    """

    def __init__(self, arms: List[float], init_std_dev: float = 1e3, alpha: float = 10,
                 n_restarts_optimizer: int = 5, normalized=True):
        """
        :param arms: list of non-normalized arms that can be pulled
        :param init_std_dev: initial value of the standard deviation for the prior
        :param alpha: value added to the diagonal of the kernel matrix during fitting. Larger values correspond to
                      increased noise level in the observations.
        :param n_restarts_optimizer: number of restarts done by the GP in order to learn the hyper-parameters
                                     of the kernel
        :param normalized: True to normalize the arms in the GP regressor, False otherwise
        """
        super().__init__(len(arms))

        self.arms: List[float] = arms
        self.gp_regressor: DiscreteGPRegressor = DiscreteGPRegressor(arms, init_std_dev, alpha, n_restarts_optimizer,
                                                                     normalized)

    def update(self, pulled_arm: int, reward: float):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.gp_regressor.fit_model(collected_rewards=self.collected_rewards,
                                    pulled_arm_history=self.pulled_arm_list)

    def pull_arm(self):
        prob_per_arm = self.gp_regressor.sample_distribution()
        return np.argmax(prob_per_arm)

    def get_optimal_arm(self) -> int:
        expected_rewards = self.gp_regressor.means
        return int(np.argmax(expected_rewards))
