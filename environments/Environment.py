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

    @staticmethod
    def get_total_time_horizon(phases: List[Phase]) -> int:
        return sum([phase.get_duration() for phase in phases])

    @staticmethod
    def get_phase_index(phases: List[Phase], time_step: int) -> int:
        duration_list = [phase.get_duration() for phase in phases]
        cum_sum_duration_list = np.cumsum(duration_list)
        idx = np.searchsorted(cum_sum_duration_list, time_step)

        return idx

    def get_n_subcampaigns(self):
        return self.n_subcampaigns

    @abstractmethod
    def round(self, *args, **kwargs):
        pass
