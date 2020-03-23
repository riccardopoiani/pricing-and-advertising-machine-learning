from typing import List, Callable
import numpy as np

from environments.Environment import Environment
from utils.stats.StochasticFunction import IStochasticFunction


class AdEnvironment(Environment):

    def __init__(self, number_of_clicks_per_budget_list: List[Callable], sigmas: List[float]):
        super(AdEnvironment, self).__init__([], [])
        self.number_of_clicks_per_budget_list = number_of_clicks_per_budget_list
        self.sigmas = sigmas

    def round(self, budget_allocation: List[float]) -> List[float]:
        rewards = [self.number_of_clicks_per_budget_list[x](budget_allocation[x]) + np.random.normal(0, self.sigmas[x])
                for x in range(len(budget_allocation))]
        return rewards
