from abc import ABC
from typing import List, Tuple

from advertising.data_structure.Campaign import Campaign
from bandit.IBandit import IBandit


class CombinatorialBandit(IBandit, ABC):
    """
    General class representing a combinatorial bandit for a specific campaign with a cumulative daily budget and
    each budget of every sub-campaign can assume values from [0, cumulative daily budget] with a uniform discretization
    """

    def __init__(self, campaign: Campaign):
        self.campaign: Campaign = campaign
        self.t: int = 0
        self.collected_rewards: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_superarm_list: List[Tuple] = []

    def update_observations(self, pulled_arm: List[float], reward: List[float]):
        """
        Update bandit statistics:
        - ordered list containing the rewards collected from the beginning of the learning process

        :param pulled_arm: the superarm that has been pulled
        :param reward: reward obtained pulling pulled_superarm
        :return: none
        """
        self.collected_rewards.append(reward)
        self.pulled_superarm_list.append(pulled_arm)


