import numpy as np
from typing import List


class Campaign(object):
    """
    Data structure in order to aggregate all the information about the campaign of an advertising problem
    """

    def __init__(self, n_sub_campaigns: int, cum_budget: int, n_arms: int):
        self._n_sub_campaigns = n_sub_campaigns
        self._budgets = np.linspace(0, cum_budget, n_arms)
        self._cum_budget = cum_budget
        self._sub_campaigns_matrix = np.zeros(shape=(n_sub_campaigns, len(self._budgets)))

    def set_sub_campaign(self, sub_campaign_idx: int, sub_campaign_values: List[float]) -> None:
        """
        Set an entire row of the sub_campaigns_matrix, which contains the value of the sub-campaign for each
        budget

        :param sub_campaign_idx: index of the sub-campaign
        :param sub_campaign_values: values of the sub-campaign
        """
        self._sub_campaigns_matrix[sub_campaign_idx] = sub_campaign_values

    def get_sub_campaigns(self) -> np.ndarray:
        """
        :return: the matrix containing all the values of sub-campaigns for each budget
        """
        return np.array(self._sub_campaigns_matrix, copy=True)

    def get_budgets(self) -> np.ndarray:
        return np.array(self._budgets, copy=True)

    def get_cum_budget(self) -> int:
        return self._cum_budget

    def get_n_sub_campaigns(self) -> int:
        return self._n_sub_campaigns

