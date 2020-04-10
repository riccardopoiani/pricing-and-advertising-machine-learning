from scipy.stats import norm
import numpy as np

from typing import List

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.joint.IJointBandit import IJointBandit


class JointBanditQuantileVisits(IJointBandit):
    """
    Quantile-based exploration for the estimated values of each sub-campaign,
    as reported in "A combinatorial-bandit algorithm for the Online joint bid/budget optimization
    of pay-per-click advertising campaign" [Nuara et. al]
    """

    def __init__(self, price_learner: List[DiscreteBandit],
                 campaign: Campaign,
                 min_std_quantile,
                 arm_values: np.array,
                 number_of_visit_model_list: List[DiscreteRegressor]):
        assert len(price_learner) == campaign.get_n_sub_campaigns()

        super().__init__(campaign=campaign)

        # Number of visit data structure
        self.number_of_visit_model_list: List[DiscreteRegressor] = number_of_visit_model_list
        self.collected_rewards_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_arm_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]

        self.price_learner: List[DiscreteBandit] = price_learner
        self.day_t = 1
        self.daily_profit = 0
        self.min_std_quantile = min_std_quantile
        self.max_expected_reward: float = arm_values.max()

    # Pull methods

    def pull_price(self, user_class: int) -> int:
        return self.price_learner[user_class].pull_arm()

    def pull_budget(self) -> List[int]:
        _, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
        return [np.where(self.campaign.get_budgets() == budget)[0][0] for budget in best_budgets]

    # Update methods

    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        self.daily_profit += observed_reward
        self.price_learner[user_class].update(pulled_arm, observed_reward)

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        sub_campaign_values = [[] for _ in range(self.campaign.get_n_sub_campaigns())]
        self.day_t += 1
        self.collected_total_rewards.append(self.daily_profit)
        self.daily_profit = 0

        # Update data structure
        for i in range(self.campaign.get_n_sub_campaigns()):
            self.pulled_arm_sub_campaign[i].append(pulled_arm_list[i])
            self.collected_rewards_sub_campaign[i].append(n_visits[i])

        # Update model with new information
        for sub_index, model in enumerate(self.number_of_visit_model_list):
            model.fit_model(collected_rewards=self.collected_rewards_sub_campaign[sub_index],
                            pulled_arm_history=self.pulled_arm_sub_campaign[sub_index])

            # Update estimations of the values of the sub-campaigns
            sub_campaign_values[sub_index] = self.number_of_visit_model_list[sub_index].sample_distribution()

        # Compute the value of the ad
        expected_rewards = [np.array(learner.collected_rewards).mean() if len(learner.collected_rewards) > 0
                            else self.max_expected_reward for learner in self.price_learner]
        std_rewards = [np.array(learner.collected_rewards).std() if len(learner.collected_rewards) > 0
                       else self.min_std_quantile for learner in self.price_learner]

        std_rewards = np.array(std_rewards)
        std_rewards = np.where(std_rewards < self.min_std_quantile, self.min_std_quantile, std_rewards)

        quantile_order = 1 - (1 / self.day_t)

        estimated_ad_value = [norm.ppf(q=quantile_order, loc=expected_rewards[i],
                                       scale=std_rewards[i]) for i in range(len(std_rewards))]

        observed_rewards = np.transpose(np.array(sub_campaign_values)) * np.array(estimated_ad_value)

        for sub_index, model in enumerate(self.number_of_visit_model_list):
            self.campaign.set_sub_campaign(sub_index, observed_rewards[:, sub_index])
