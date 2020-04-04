import numpy as np

from abc import ABC
from typing import List

from advertising.data_structure import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.IBandit import IBandit


class CombinatorialBandit(IBandit, ABC):
    """
    General class for the combinatorial bandit for a specific campaign with a cumulative daily budget and
    each budget of every sub-campaign can assume values from [0, cumulative daily budget] with a uniform discretization
    """

    def __init__(self, campaign: Campaign, model_list: List[DiscreteRegressor]):
        """
        :param campaign: campaign that will be optimized
        :param model_list: regressors that will be used to estimate the quantity in each campaign. There is a regressor
        for each campaign
        """
        assert len(model_list) == campaign.get_n_sub_campaigns()

        self.campaign: Campaign = campaign
        self.t: int = 0
        self.collected_rewards: List[float] = []
        self.pulled_superarm_list: List[List] = []
        self.model_list: List[DiscreteRegressor] = model_list

        self.collected_rewards_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_arm_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]

        for sub_index, model in enumerate(self.model_list):
            sub_campaign_values = self.model_list[sub_index].sample_distribution()
            self.campaign.set_sub_campaign(sub_index, sub_campaign_values)

    def pull_arm(self) -> List[int]:
        """
        Find the best allocation of budgets by optimizing the combinatorial problem of the campaign and then return
        the indices of the best budgets.
        The combinatorial problem is optimized given estimates provided only by the last data (amount specified
        by the sliding window)

        :return: the indices of the best budgets given the actual campaign
        """
        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
        return [np.where(self.campaign.get_budgets() == budget)[0][0] for budget in best_budgets]


