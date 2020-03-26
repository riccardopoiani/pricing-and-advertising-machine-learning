import math
from typing import List, Callable
import numpy as np

from environments.Settings.Scenario import Scenario
from utils.stats.StochasticFunction import IStochasticFunction, BoundedLambdaStochasticFunction


class PolynomialAdvertisingScenario(Scenario):
    @staticmethod
    def get_scenario(linear_coefficients: List[float], exponents: List[float], biases: List[float],
                     max_n_clicks_list: List[int], n_clicks_function_std: List[float], cum_budget: float,
                     n_subcampaigns: int = 3) -> (List[IStochasticFunction], List[IStochasticFunction]):
        assert len(linear_coefficients) == n_subcampaigns
        assert len(exponents) == n_subcampaigns
        assert len(biases) == n_subcampaigns
        assert len(max_n_clicks_list) == n_subcampaigns
        assert len(n_clicks_function_std) == n_subcampaigns

        def generate_n_clicks_function(coeff: float, exp: float, bias: float, max_n_clicks: float, n_clicks_std: float):
            return lambda budget: int(np.random.normal(min((budget * coeff) ** exp + bias, max_n_clicks),
                                                       n_clicks_std))

        functions_lambda = [
            generate_n_clicks_function(linear_coefficients[i], exponents[i], biases[i], max_n_clicks_list[i],
                                       n_clicks_function_std[i])
            for i in range(len(linear_coefficients))
        ]

        n_clicks_function: List[IStochasticFunction] = [BoundedLambdaStochasticFunction(f=functions_lambda[i])
                                                        for i in range(n_subcampaigns)]
        return [], n_clicks_function

    @staticmethod
    def get_scenario_name() -> str:
        return "POLYNOMIAL_ADVERTISING_SCENARIO"
