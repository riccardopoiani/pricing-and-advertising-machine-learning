import numpy as np
from typing import List


class Campaign(object):

    def __init__(self, n_sub_campaigns: int, cum_budget: int, budgets: List[int]):
        self._sub_campaigns_matrix = np.zeros(shape=(n_sub_campaigns, len(budgets)))
        self._n_sub_campaigns = n_sub_campaigns
        self._budgets = budgets
        self._cum_budget = cum_budget

    def set_sub_campaign(self, sub_campaign_idx: int, sub_campaign_values: List[float]):
        self._sub_campaigns_matrix[sub_campaign_idx] = sub_campaign_values

    def get_sub_campaigns(self) -> np.ndarray:
        return np.array(self._sub_campaigns_matrix, copy=True)

    def get_budgets(self):
        return np.array(self._budgets, copy=True)

    def get_cum_budget(self):
        return self._cum_budget

    def get_n_sub_campaigns(self):
        return self._n_sub_campaigns

