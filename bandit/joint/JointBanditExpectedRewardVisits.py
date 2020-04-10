import numpy as np

from typing import List

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from bandit.discrete.DiscreteBandit import DiscreteBandit
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.joint.IJointBandit import IJointBandit


class JointBanditExpectedRewardVisits(IJointBandit):
    """
    General class for the problem of jointly optimizing the pricing and the advertising strategy,
    based on wrapping a combinatorial bandit for advertising and multiple bandits for learning
    the price for each sub-campaign.
    """

    def __init__(self, price_learner: List[DiscreteBandit],
                 campaign: Campaign,
                 arm_values: np.array,
                 number_of_visit_model_list: List[DiscreteRegressor]):
        assert len(number_of_visit_model_list) == campaign.get_n_sub_campaigns()
        assert len(price_learner) == campaign.get_n_sub_campaigns()

        super().__init__(campaign=campaign)

        # Number of visit data structure
        self.number_of_visit_model_list: List[DiscreteRegressor] = number_of_visit_model_list
        self.collected_rewards_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_arm_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]

        self.price_learner: List[DiscreteBandit] = price_learner
        self.daily_profit = 0
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

        expected_rewards = [np.array(learner.collected_rewards).mean() if len(learner.collected_rewards) > 0
                            else self.max_expected_reward for learner in self.price_learner]
        observed_rewards = np.transpose(np.array(sub_campaign_values)) * np.array(expected_rewards)

        for sub_index, model in enumerate(self.number_of_visit_model_list):
            self.campaign.set_sub_campaign(sub_index, observed_rewards[:, sub_index])
