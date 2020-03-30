from abc import ABC
from typing import List

from advertising.data_structure import Campaign
from advertising.regressors import DiscreteRegressor
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
