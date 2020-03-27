import numpy as np

from abc import ABC, abstractmethod
from typing import List

from utils.stats.StochasticFunction import IStochasticFunction, BoundedLambdaStochasticFunction


class Scenario(ABC):
    """
    General scenario for the experiments, defining the conversion rate probability and
    the number of visits
    """

    @staticmethod
    @abstractmethod
    def get_scenario(**kwargs) -> (List[IStochasticFunction], List[IStochasticFunction]):
        """
        :param kwargs: arguments for creating the conversion rate probabilities and the number of visits
        :return: (conversion rate probabilities, number of visits)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scenario_name() -> str:
        """
        :return: name of the scenario
        """
        pass


class LinearPriceLinearClickScenario(Scenario):

    @staticmethod
    def get_scenario_name() -> str:
        return "LINEAR_PRICE_LINEAR_CLICK"

    @staticmethod
    def get_scenario(price_multiplier: List, max_price: List, min_price: List, budget_multiplier: List,
                     budget_std: List, max_n_clicks_list: List,
                     n_subcampaigns: int = 3) -> (List[IStochasticFunction], List[IStochasticFunction]):
        """
        This setting considers the following scenario:
        - There are 3 classes of users
        - Conversion rate probabilities are linear decreasing bernoullian function, that considers a certain range
        [min_value, max_value] for the price. The format of this function is 1 - price_multiplier[i] * price / (max_price[i]*2)
        - Number of clicks are functions of a upper bounded linear function with a certain coefficient
          "budget_multiplier" and a certain noised. The upper bound of each number of click function is defined in
          "max_n_clicks_list"

        All provided lists must be of the same dimension of the specified number of classes

        :param n_subcampaigns: number of subcampaigns equal to the number of classes
        :param max_n_clicks_list: number of maximum clicks of a certain class of users
        :param budget_std: standard deviation of the gaussian clipped distribution of the number of users
        :param budget_multiplier: scale of the gaussian clipped distribution of the number of users
        :param min_price: minimum possible price for the stochastic function of the conversion rate probability
        :param max_price: maximum possible price for the stochastic function of the conversion rate probability
        :param price_multiplier: multiplier of the price for the stochastic function of the conversion rate probability

        :return: (list of conversion rate probabilities, list of number of clicks probabilities)
        """
        assert len(price_multiplier) == n_subcampaigns
        assert len(max_price) == n_subcampaigns
        assert len(min_price) == n_subcampaigns
        assert len(budget_multiplier) == n_subcampaigns
        assert len(budget_std) == n_subcampaigns
        assert len(max_n_clicks_list) == n_subcampaigns

        # Conversion rate probabilities
        crp_lambda = [lambda price: np.random.binomial(n=1, p=1 - price_multiplier[i] * price / max_price[i] * 2)
                      for i in range(n_subcampaigns)]
        crp: List[IStochasticFunction] = [BoundedLambdaStochasticFunction(min_value=min_price[i],
                                                                          max_value=max_price[i],
                                                                          f=crp_lambda[i])
                                          for i in range(n_subcampaigns)]

        # Number of clicks
        def generate_n_clicks_function(coeff: float, max_n_clicks: float, n_clicks_std: float):
            return lambda budget: int(np.random.normal(min((budget * coeff), max_n_clicks),
                                                       n_clicks_std))

        functions_lambda = [
            generate_n_clicks_function(budget_multiplier[i], max_n_clicks_list[i], budget_std[i])
            for i in range(n_subcampaigns)
        ]

        n_clicks: List[IStochasticFunction] = [BoundedLambdaStochasticFunction(f=functions_lambda[i])
                                               for i in range(n_subcampaigns)]

        return crp, n_clicks
