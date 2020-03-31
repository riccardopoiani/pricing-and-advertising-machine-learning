from typing import List

from environments.Environment import Environment
from environments.Settings.Phase import Phase


class AdvertisingEnvironment(Environment):
    """
    Environment for an advertising scenario with no abrupt phases
    """

    def __init__(self, scenario):
        super(AdvertisingEnvironment, self).__init__(scenario)

    def round(self, budget_allocation: List[float]) -> List[float]:
        """
        Simulate a seller allocating a budget for the day

        :param budget_allocation: list of budgets to be allocated on every subcampaigns
        :return: the list of rewards for each subcampaign
        """
        self.day_t += 1

        current_phase = self.get_current_phase()
        rewards = current_phase.get_all_n_clicks(budget_allocation=budget_allocation)
        return rewards
