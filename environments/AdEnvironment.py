from typing import Callable
import numpy as np


class AdEnvironment(object):

    def __init__(self, number_of_clicks_fun: Callable[[np.ndarray], np.ndarray], budgets: np.ndarray, std_dev: float):
        """

        :param number_of_clicks_fun:
        :param budgets: np.array
        :param std_dev:
        """
        self.budgets = budgets
        self.means = number_of_clicks_fun(budgets)
        self.sigmas = np.ones(len(budgets))*std_dev

    def round(self, pulled_arm: int) -> float:
        """

        :param pulled_arm: "index"
        :return:
        """
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
