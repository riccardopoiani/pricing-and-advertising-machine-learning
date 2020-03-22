from abc import ABC, abstractmethod
from typing import List

from utils.stats.StochasticFunction import IStochasticFunction


class Environment(ABC):
    """
    General environment for carrying out simulations
    """

    def __init__(self, conversion_rate_probabilities: List[IStochasticFunction],
                 number_of_visit: List[IStochasticFunction]):
        self.crp_list: List[IStochasticFunction] = conversion_rate_probabilities
        self.number_of_visit_list: List[IStochasticFunction] = number_of_visit

    @abstractmethod
    def round(self, *args, **kwargs):
        pass
