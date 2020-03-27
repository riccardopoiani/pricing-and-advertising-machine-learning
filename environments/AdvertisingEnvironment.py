from typing import List

from environments.Environment import Environment
from environments.Phase import Phase


class AdvertisingEnvironment(Environment):

    def __init__(self, n_subcampaigns, phases: List[Phase]):
        super(AdvertisingEnvironment, self).__init__(n_subcampaigns, phases)

    def round(self, budget_allocation: List[float]) -> List[float]:
        self.day_t += 1

        phase_idx = self.get_phase_index(self.phases, self.day_t)
        rewards = self.phases[phase_idx].get_all_n_clicks(budget_allocation=budget_allocation)
        return rewards
