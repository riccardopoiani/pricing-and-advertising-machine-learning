from abc import ABC, abstractmethod
from typing import List

from advertising.data_structure.Campaign import Campaign


class IJointBandit(ABC):
    """
    General class for enabling the joint optimization of the advertising and pricing strategy
    """

    def __init__(self, campaign: Campaign):
        self.campaign = campaign
        self.collected_total_rewards: List[float] = []
        self.daily_visit: List[List[int]] = [[] for _ in range(campaign.get_n_sub_campaigns())]

    @abstractmethod
    def pull_price(self, user_class: int) -> int:
        """
        Price the user, knowing its context

        :param user_class: index of the user context
        :return: index of the chosen price for the given user
        """
        pass

    @abstractmethod
    def pull_budget(self) -> List[int]:
        """
        :return: budget to be spent in each sub-campaign
        """
        pass

    @abstractmethod
    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        """
        Update the price learner for a given user class, considering an observation

        :param user_class class of the user
        :param pulled_arm arm that has been pulled for that user
        :param observed_reward reward that has been observed while pulling a certain arm
        :return: None
        """
        pass

    @abstractmethod
    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        """
        Update the advertising learner given that budget for each sub-campaign and the number of visits that
        they have generated during the day, for each target

        :param pulled_arm_list: used budget for each sub-campaign
        :param n_visits: number of visit generated for each sub-campaign
        :return: None
        """
        pass

    def get_daily_number_of_visit_per_sub_campaign(self) -> List[List[int]]:
        """
        :return: a list containing a list for each sub-campaign. Each of this
        list contain the number of visit generated due to that sub-campaign
        """
        return self.daily_visit

    def get_daily_reward(self) -> List[float]:
        """
        :return: a list containing the daily reward of the joint optimization of
        the pricing and the advertising strategies
        """
        return self.collected_total_rewards
