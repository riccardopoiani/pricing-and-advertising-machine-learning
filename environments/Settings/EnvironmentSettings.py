<<<<<<< Updated upstream
import numpy as np

from typing import List, Tuple
=======
from typing import List
>>>>>>> Stashed changes

from environments.Settings.AdvertisingScenario import PolynomialAdvertisingScenario
from environments.Settings.Scenario import Scenario, LinearPriceGaussianVisitsScenario
from utils.stats.StochasticFunction import IStochasticFunction


class EnvironmentManager(object):

    scenario_list: List[Scenario] = [LinearPriceGaussianVisitsScenario, PolynomialAdvertisingScenario]

    @staticmethod
    def get_setting(setting_name, **kwargs) -> Tuple:
        """
        Retrieve a setting on the basis of its name

        :param setting_name: name of the setting
        :param kwargs: parameters for the desired setting
        :return: (conversion rate probability and number of visits)
        """
        for s in EnvironmentManager.scenario_list:
            if s.get_scenario_name() == setting_name:
                return s.get_scenario(**kwargs)
