from typing import List

from environments.Settings.AdvertisingScenario import PolynomialAdvertisingScenario
from environments.Settings.Scenario import Scenario, LinearPriceLinearClickScenario
from utils.stats.StochasticFunction import IStochasticFunction


class EnvironmentManager(object):

    scenario_list: List[Scenario] = [LinearPriceLinearClickScenario, PolynomialAdvertisingScenario]

    @staticmethod
    def get_setting(setting_name, **kwargs) -> (List[IStochasticFunction], List[IStochasticFunction]):
        """
        Retrieve a setting on the basis of its name

        :param setting_name: name of the setting
        :param kwargs: parameters for the desired setting
        :return: (conversion rate probability and number of visits)
        """
        for s in EnvironmentManager.scenario_list:
            if s.get_scenario_name() == setting_name:
                return s.get_scenario(**kwargs)
