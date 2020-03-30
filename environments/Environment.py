from abc import abstractmethod, ABC
from typing import List

import numpy as np

from environments.Phase import Phase


class Environment(ABC):
    """
    General environment for carrying out simulations

    Assumptions:
     - the number of subcampaigns remains constant through all the phases
     - the classes of users are equivalent to the subcampaigns
     - the number of phases of pricing is equal to the number of phases of advertisement
     - the time horizon of the pricing and advertisement has to be the same
    """

    def __init__(self, n_subcampaigns: int, phases: List[Phase]):
        # Assertions
        for phase in phases:
            assert n_subcampaigns == phase.get_n_subcampaigns()

        self.day_t = 0
        self.n_subcampaigns = n_subcampaigns
        self.phases: List[Phase] = phases

    def get_total_time_horizon(self) -> int:
        """
        :return: the total time horizon of the phases
        """
        return sum([phase.get_duration() for phase in self.phases])

    def get_current_phase(self) -> Phase:
        """
        :return: the index of the actual phase correlated to the time step given
        """
        duration_list = [phase.get_duration() for phase in self.phases]
        cum_sum_duration_list = np.cumsum(duration_list)
        idx = np.searchsorted(cum_sum_duration_list, self.day_t)
        return self.phases[idx]

    def get_n_subcampaigns(self):
        return self.n_subcampaigns

    @abstractmethod
    def round(self, *args, **kwargs):
        pass
