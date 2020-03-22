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


class LinearPriceGaussianVisitsScenario(Scenario):

    @staticmethod
    def get_scenario_name() -> str:
        return "LINEAR_PRICE_GAUSSIAN_VISIT"

    @staticmethod
    def get_scenario(price_multiplier: List, max_price: List, min_price: List, budget_scale: List,
                     budget_std: List, min_budget: List, max_budget: List, min_n_visit: List,
                     max_n_visit: List, n_classes=3) -> (List[IStochasticFunction], List[IStochasticFunction]):
        """
        This setting considers the following scenario:
        - There are 3 classes of users
        - Conversion rate probabilities are linear decreasing bernoullian function, that considers a certain range
        [min_value, max_value] for the price. The format of this function is 1 - price_multiplier[i] * price / (max_price[i]*2)
        - Number of visit probabilities are function of a clipped gaussian distribution
        with a certain mean and a certain standard deviation. In particular, as mean budget * budget_scale[i] is considered
        and the standard deviation is fixed among for every budget value. The clip is performed with a minimum and a maximum
        number of users specified with min_n_visit[i] and max_n_visit[i] (if None is given, no clipping is performed).
        Moreover, the budget is delimited between min_budget[i] and max_budget[i]

        All provided lists must be of the same dimension of the specified number of classes

        :param n_classes: number of user classes
        :param max_n_visit: number of maximum visit of a certain class of users
        :param min_n_visit: number of minimum visit of a certain class of users
        :param budget_std: standard deviation of the gaussian clipped distribution of the number of users
        :param budget_scale: scale of the gaussian clipped distribution of the number of users
        :param min_budget: minimum possible budget for the stochastic function of the number of visits
        :param max_budget: maximum possible budget for the stochastic function of the number of visits
        :param min_price: minimum possible price for the stochastic function of the conversion rate probability
        :param max_price: maximum possible price for the stochastic function of the conversion rate probability
        :param price_multiplier: multiplier of the price for the stochastic function of the conversion rate probability

        :return: (list of conversion rate probabilities, list of number of visit probabilities)
        """
        # Conversion rate probabilities
        assert len(price_multiplier) == n_classes
        assert len(max_price) == n_classes
        assert len(min_price) == n_classes
        assert len(budget_scale) == n_classes
        assert len(budget_std) == n_classes
        assert len(min_budget) == n_classes
        assert len(max_budget) == n_classes
        assert len(min_n_visit) == n_classes
        assert len(max_n_visit) == n_classes

        crp_lambda = [lambda price: np.random.binomial(n=1, p=1 - price_multiplier[i] * price / max_price[i] * 2)
                      for i in range(n_classes)]
        crp: List[IStochasticFunction] = [BoundedLambdaStochasticFunction(min_value=min_price[i],
                                                                          max_value=max_price[i],
                                                                          f=crp_lambda[i])
                                          for i in range(n_classes)]

        # Number of visits probabilities
        nov_lambda = [
            lambda budget: lambda budget: np.clip(int(np.random.normal(budget * budget_scale[i], budget_std[i])),
                                                  min_budget[i], max_budget[i])
            for i in range(n_classes)]
        nov: List[IStochasticFunction] = [BoundedLambdaStochasticFunction(min_value=min_n_visit[i],
                                                                          max_value=max_n_visit[i],
                                                                          f=nov_lambda[i])
                                          for i in range(n_classes)]

        return crp, nov
