from abc import abstractmethod
from typing import List

import numpy as np
from scipy.stats import norm


class AdValueStrategy(object):

    @abstractmethod
    def get_estimated_value_per_clicks(self, rewards_per_subcampaign: List[List[float]], day_t: int) -> List[float]:
        pass


class QuantileAdValueStrategy(AdValueStrategy):
    """
    Quantile-based exploration for the estimated values of each sub-campaign,
    as reported in "A combinatorial-bandit algorithm for the Online joint bid/budget optimization
    of pay-per-click advertising campaign" [Nuara et. al]
    """

    def __init__(self, max_mean, min_std):
        self.max_mean = max_mean
        self.min_std = min_std

    def get_estimated_value_per_clicks(self, rewards_per_subcampaign, day_t):
        quantile_order = 1 - (1 / (day_t + 1))

        expected_rewards = [np.mean(rewards) if len(rewards) > 0 else self.max_mean
                            for rewards in rewards_per_subcampaign]
        std_rewards = np.array([np.std(rewards) if len(rewards) > 0 else self.min_std
                                for rewards in rewards_per_subcampaign])

        std_rewards = np.where(std_rewards < self.min_std, self.min_std, std_rewards)

        return [norm.ppf(q=quantile_order, loc=expected_rewards[i], scale=std_rewards[i])
                for i in range(len(std_rewards))]


class ExpectationAdValueStrategy(AdValueStrategy):
    """
    Estimation of the value per clicks by expected rewards w/o considering any uncertainty
    """

    def __init__(self, max_mean):
        self.max_mean = max_mean

    def get_estimated_value_per_clicks(self, rewards_per_subcampaign, day_t):
        expected_rewards = [np.mean(rewards) if len(rewards) > 0 else self.max_mean
                            for rewards in rewards_per_subcampaign]
        return expected_rewards
