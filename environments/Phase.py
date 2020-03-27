from abc import ABC
from typing import List

from utils.stats.StochasticFunction import IStochasticFunction


class Phase(object):
    """
    Abstract class representing a generic abrupt phase of the experiments
    """

    def __init__(self, duration: int, n_clicks_functions: List[IStochasticFunction],
                 crp_functions: List[IStochasticFunction]):
        """
        :param duration: time duration of the phase in "round"
        """
        assert len(crp_functions) == len(n_clicks_functions)

        self._duration = duration
        self._n_clicks_functions = n_clicks_functions
        self._crp_functions = crp_functions

    def get_duration(self):
        return self._duration

    def get_n_subcampaigns(self):
        return len(self._crp_functions)

    def get_all_n_clicks(self, budget_allocation: List[float]) -> List[float]:
        return [self._n_clicks_functions[idx].draw_sample(budget_allocation[idx])
                for idx in range(len(self._n_clicks_functions))]

    def get_n_clicks(self, sub_idx, budget):
        return self._n_clicks_functions[sub_idx].draw_sample(budget)

    def get_all_crp(self, price: List[float]) -> List[float]:
        return [self._crp_functions[idx].draw_sample(price[idx])
                for idx in range(len(self._crp_functions))]

    def get_crp(self, class_idx, price):
        return self._crp_functions[class_idx].draw_sample(price)
