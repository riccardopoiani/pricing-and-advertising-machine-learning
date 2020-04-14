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

    def get_all_n_clicks_sample(self, budget_allocation: List[float]) -> List[float]:
        """
        :param budget_allocation: list of budgets that are allocated
        :return: the list of number of clicks for each subcampaign
        """
        return [self._n_clicks_functions[idx].draw_sample(budget_allocation[idx])
                for idx in range(len(self._n_clicks_functions))]

    def get_n_clicks_sample(self, sub_idx: int, budget: float) -> float:
        """
        :param sub_idx: the subcampaign index
        :param budget: the chosen budget
        :return: the number of clicks for the given subcampaign and budget
        """
        return self._n_clicks_functions[sub_idx].draw_sample(budget)

    def get_all_crp_sample(self, price: List[float]) -> List[float]:
        """
        :param price: the list of prices chosen for each conversion rate probability
        :return: the list of conversion rate probability for each class (equivalent to number of subcampaigns)
        """
        return [self._crp_functions[idx].draw_sample(price[idx])
                for idx in range(len(self._crp_functions))]

    def get_crp_sample(self, class_idx: int, price: float) -> float:
        """
        :param class_idx: the class index
        :param price: the chosen price
        :return: the conversion rate probability for the given class and price
        """
        return self._crp_functions[class_idx].draw_sample(price)

    def get_crp_function(self):
        return self._crp_functions

    def get_n_clicks_function(self):
        return self._n_clicks_functions
