from scipy.stats import norm
import numpy as np

from typing import List

from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.JointBanditWrapper import JointBanditWrapper


class JointBanditQuantile(JointBanditWrapper):
    """
    Quantile-based exploration for the estimated values of each sub-campaign,
    as reported in "A combinatorial-bandit algorithm for the Online joint bid/budget optimization
    of pay-per-click advertising campaign" [Nuara et. al]
    """

    def __init__(self, ads_learner: CombinatorialBandit,
                 price_learner: List[DiscreteBandit],
                 campaign: Campaign):
        super().__init__(ads_learner=ads_learner,
                         price_learner=price_learner,
                         campaign=campaign)
        self.day_t = 1

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        self.day_t += 1
        # Compute the value of the ad
        expected_rewards = [np.array(learner.collected_rewards).mean() if len(learner.collected_rewards) > 0
                            else 0 for learner in self.price_learner]
        std_rewards = [np.array(learner.collected_rewards).std() if len(learner.collected_rewards) > 0
                       else 1 for learner in self.price_learner]

        quantile_order = 1 - (1 / self.day_t)

        estimated_ad_value = [norm.ppf(q=quantile_order, loc=expected_rewards[i],
                                       scale=std_rewards[i]) for i in range(len(std_rewards))]

        observed_rewards = np.array(n_visits) * np.array(estimated_ad_value)

        # Update the model
        self.ads_learner.update(pulled_arm=pulled_arm_list, observed_reward=observed_rewards)

        # Update the current reward obtained from the model
        total_r = np.array([np.array(learner.collected_rewards).sum() for learner in self.price_learner]).sum()
        self.collected_total_rewards.append(total_r - np.array(self.collected_total_rewards).sum())