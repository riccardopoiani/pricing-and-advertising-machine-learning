import math
from typing import List, Callable

from environments.Settings.Scenario import Scenario


class PolynomialAdvertisingScenario(Scenario):
    @staticmethod
    def get_scenario(linear_coefficients: List[float], exponents: List[float], biases: List[float],
                     max_n_clicks_list: List[float]) -> List[Callable]:
        number_of_clicks_per_budget = [(lambda coeff, exp, bias, max_n_clicks:
                                        (lambda budget: min(coeff * budget ** exp + bias, max_n_clicks)))
                                       (coeff, exp, bias, max_n_clicks)
                                       for coeff, exp, bias, max_n_clicks in
                                       zip(linear_coefficients, exponents, biases, max_n_clicks_list)]
        return number_of_clicks_per_budget

    @staticmethod
    def get_scenario_name() -> str:
        return "POLYNOMIAL_ADVERTISING_SCENARIO"


class LogarithmicAdvertisingScenario(Scenario):
    @staticmethod
    def get_scenario(linear_coefficients: List[float], biases: List[float]) -> List[Callable]:
        number_of_clicks_per_budget = []
        for coeff, bias in zip(linear_coefficients, biases):
            number_of_clicks_per_budget.append(lambda budget: coeff * math.log(budget) + bias)
        return number_of_clicks_per_budget

    @staticmethod
    def get_scenario_name() -> str:
        return "LOGARITHMIC_ADVERTISING_SCENARIO"
