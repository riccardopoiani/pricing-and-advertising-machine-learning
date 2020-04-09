from scipy.stats import norm
import numpy as np

from typing import List

from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.IJointBandit import IJointBandit


class JointBanditQuantile(IJointBandit):
    """
    Quantile-based exploration for the estimated values of each sub-campaign,
    as reported in "A combinatorial-bandit algorithm for the Online joint bid/budget optimization
    of pay-per-click advertising campaign" [Nuara et. al]
    """

    def __init__(self, ads_learner: CombinatorialBandit,
                 price_learner: List[DiscreteBandit],
                 campaign: Campaign,
                 min_std_quantile):
        assert len(price_learner) == campaign.get_n_sub_campaigns()

        super().__init__(campaign=campaign)
        self.ads_learner: CombinatorialBandit = ads_learner
        self.price_learner: List[DiscreteBandit] = price_learner
        self.day_t = 1
        self.daily_profit = 0
        self.min_std_quantile = min_std_quantile

    # Pull methods

    def pull_price(self, user_class: int) -> int:
        return self.price_learner[user_class].pull_arm()

    def pull_budget(self) -> List[int]:
        return self.ads_learner.pull_arm()

    # Update methods

    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        self.daily_profit += observed_reward
        self.price_learner[user_class].update(pulled_arm, observed_reward)

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        self.day_t += 1
        self.collected_total_rewards.append(self.daily_profit)
        self.daily_profit = 0

        # Compute the value of the ad
        expected_rewards = [np.array(learner.collected_rewards).mean() if len(learner.collected_rewards) > 0
                            else 0 for learner in self.price_learner]
        std_rewards = [np.array(learner.collected_rewards).std() if len(learner.collected_rewards) > 0
                       else 1 for learner in self.price_learner]

        std_rewards = np.array(std_rewards)
        std_rewards = np.where(std_rewards < self.min_std_quantile, self.min_std_quantile, std_rewards)

        quantile_order = 1 - (1 / self.day_t)

        estimated_ad_value = [norm.ppf(q=quantile_order, loc=expected_rewards[i],
                                       scale=std_rewards[i]) for i in range(len(std_rewards))]

        observed_rewards = np.array(n_visits) * np.array(estimated_ad_value)

        # Update the model
        self.ads_learner.update(pulled_arm=pulled_arm_list, observed_reward=observed_rewards)
