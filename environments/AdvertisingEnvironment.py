from typing import List

from environments.Environment import Environment
from utils.stats.StochasticFunction import IStochasticFunction


class AdvertisingEnvironment(Environment):

    def __init__(self, n_clicks_list: List[IStochasticFunction]):
        super(AdvertisingEnvironment, self).__init__([], n_clicks_list)

    def round(self, budget_allocation: List[float]) -> List[float]:
        rewards = [self.number_of_visit_list[budget_idx].draw_sample(budget_allocation[budget_idx])
                   for budget_idx in range(len(budget_allocation))]
        return rewards
