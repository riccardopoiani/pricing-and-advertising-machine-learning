from typing import List, Tuple

import numpy as np

from environments.Environment import Environment
from environments.Settings.Scenario import Scenario


class PricingEnvironmentContextGeneration(Environment):
    """
    Environment for the pricing scenario
    """

    def __init__(self, scenario: Scenario, fixed_budget_allocation: List[float]):
        """
        Create a pricing stationary environment with a fixed budget

        :param fixed_budget_allocation: fixed budget allocation to be used in order to simulate the number of visits
        """
        assert len(fixed_budget_allocation) == scenario.get_n_subcampaigns()

        super().__init__(scenario)
        self.sampled_users: List[Tuple[int, ...]] = []
        self.fixed_budget_allocation: List[float] = fixed_budget_allocation
        self.current_phase = self.get_current_phase()
        self.day_breakpoints: List[int] = []
        self.user_count = 0

    def round(self, price, min_context):
        """
        Simulate if a user buys or not the product

        :param price: price used for the user
        :param min_context: min context specifying the user
        :return: 1 if the users has bought the item, 0 otherwise
        """
        self.user_count += 1
        user_class: int = self.scenario.get_min_context_to_subcampaign_dict().get(min_context)
        return self.current_phase.get_crp(class_idx=user_class, price=price)

    def get_day_breakpoints(self) -> List[int]:
        return self.day_breakpoints

    def get_number_of_days(self) -> int:
        return self.day_t

    def get_n_users(self) -> int:
        self.current_phase = self.get_current_phase()
        n_users_list = self.current_phase.get_all_n_clicks(budget_allocation=self.fixed_budget_allocation)
        for i in range(len(n_users_list)):
            min_contexts = self.scenario.get_min_contexts_for_subcampaign(i)
            for min_context in min_contexts:
                self.sampled_users.extend([min_context] * int((n_users_list[i]) / len(min_contexts)))
        np.random.shuffle(self.sampled_users)
        self.day_t += 1
        self.day_breakpoints.append(len(self.sampled_users))
        return len(self.sampled_users)

    def get_user(self) -> Tuple[int, ...]:
        user = self.sampled_users.pop()
        return user
