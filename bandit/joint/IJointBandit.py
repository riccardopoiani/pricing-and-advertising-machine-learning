from abc import ABC, abstractmethod
from typing import List


class IJointBandit(ABC):
    """
    General class for enabling the joint optimization of the advertising and pricing strategy
    """

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