from typing import List

import numpy as np

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit


class CombinatorialStationaryBandit(CombinatorialBandit):
    """
    Class representing a stationary combinatorial bandit
    """

    def __init__(self, campaign: Campaign, model_list: List[DiscreteRegressor]):
        super().__init__(campaign=campaign, model_list=model_list)

    def pull_arm(self) -> List[int]:
        """
        Find the best allocation of budgets by optimizing the combinatorial problem of the campaign and then return
        the indices of the best budgets

        :return: the indices of the best budgets given the actual campaign
        """
        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
        return [np.where(self.campaign.get_budgets() == budget)[0][0] for budget in best_budgets]

    def update_observations(self, pulled_arm: List[int], reward: List[float]) -> None:
        """
        Update the combinatorial bandit statistics:
         - ordered list containing the rewards collected since the beginning
         - ordered list of the superarm pulled since the beginning

        :param pulled_arm: the superarm that has been pulled (list of indices)
        :param reward: reward obtained pulling pulled_superarm
        :return: None
        """
        self.collected_rewards.append(sum(reward))
        self.pulled_superarm_list.append(pulled_arm)

        for i in range(self.campaign.get_n_sub_campaigns()):
            self.pulled_arm_sub_campaign[i].append(pulled_arm[i])
            self.collected_rewards_sub_campaign[i].append(reward[i])

    def update(self, pulled_arm: List[int], reward: List[float]) -> None:
        """
        Update observations and models of the sub-campaign

        :param pulled_arm: list of indices of the pulled arms (i.e. superarm pulled)
        :param reward: list of observed reward for each pulled arm
        :return: None
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        for sub_index, model in enumerate(self.model_list):
            model.fit_model(collected_rewards=self.collected_rewards_sub_campaign[sub_index],
                            pulled_arm_history=self.pulled_arm_sub_campaign[sub_index])

            # Update estimations of the values of the sub-campaigns
            sub_campaign_values = self.model_list[sub_index].sample_distribution()
            self.campaign.set_sub_campaign(sub_index, sub_campaign_values)

    def get_optimal_arm(self) -> int:
        # TODO
        pass
