from typing import List

import numpy as np

from environments.Environment import Environment
from environments.Phase import Phase
from typing import List


class PricingEnvironmentFixedBudget(Environment):
    """
    Environment for the pricing scenario
    """

    def __init__(self, n_subcampaigns: int, phases: List[Phase], fixed_budget_allocation: List[float]):
        """
        Create a pricing stationary environment with a fixed budget

        :param fixed_budget_allocation: fixed budget allocation to be used in order to simulate the number of visits
        """
        assert len(fixed_budget_allocation) == n_subcampaigns

        super().__init__(n_subcampaigns, phases)
        self.sampled_users: List = []
        self.fixed_budget_allocation = fixed_budget_allocation
        self.current_user_class = 0
        self.current_phase = 0
        self.day_breakpoints: List[int] = []
        self.user_count = 0

    def round(self, price):
        """
        Simulate if a user buys or not the product

        :param price: price used for the user
        :return: 1 if the users has bought the item, 0 otherwise
        """
        self.user_count += 1
        self.simulate_user_arrive()
        return self.phases[self.current_phase].get_crp(class_idx=self.current_user_class, price=price)

    def simulate_user_arrive(self) -> None:
        """
        Simulate the arrival of a user on the website: in particular, it fills the array of sampled users
        determining the class of the current user that has arrived.
        In the case in which no more users are available, a day has elapsed and the users are re-sampled
        for the next day.
        """
        # Check if the day is over: in this case, samples users from the number of visits distribution
        self.current_phase = self.get_phase_index(self.phases, self.day_t)

        elapsed_day = False
        while len(self.sampled_users) == 0:
            n_users_list = self.phases[self.current_phase].\
                get_all_n_clicks(budget_allocation=self.fixed_budget_allocation)
            for i in range(len(n_users_list)):
                self.sampled_users.extend([i] * int(n_users_list[i]))
            np.random.shuffle(self.sampled_users)
            elapsed_day = True

        if elapsed_day:
            self.day_t += 1
            self.day_breakpoints.append(self.user_count)

        self.current_user_class = self.sampled_users.pop()

    def get_day_breakpoints(self) -> List[int]:
        return self.day_breakpoints

    def get_number_of_days(self) -> int:
        return self.day_t
