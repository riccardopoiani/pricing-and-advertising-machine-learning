from typing import List

from environments.Environment import Environment
from environments.Phase import Phase


class AdvertisingEnvironment(Environment):
    """
    Environment for an advertising scenario with no abrupt phases
    """

    def __init__(self, n_subcampaigns: int, phases: List[Phase]):
        super(AdvertisingEnvironment, self).__init__(n_subcampaigns, phases)

    def round(self, budget_allocation: List[float]) -> List[float]:
        """
        Simulate a seller allocating a budget for the day

        :param budget_allocation: list of budgets to be allocated on every subcampaigns
        :return: the list of rewards for each subcampaign
        """
        self.day_t += 1

        phase_idx = self.get_phase_index(self.phases, self.day_t)
        rewards = self.phases[phase_idx].get_all_n_clicks(budget_allocation=budget_allocation)
        return rewards
