import numpy as np

from typing import List, Tuple

from environments.Environment import Environment
from environments.Settings.Scenario import Scenario


class PricingAdvertisingJointEnvironment(Environment):
    """
    Environment for jointly optimizing the pricing and the advertising strategy
    """

    def __init__(self, scenario: Scenario):
        super().__init__(scenario=scenario)

        # Current data
        self.sampled_users: List[Tuple[int, ...]] = []
        self.daily_users: List[int] = []
        self.budget_allocation: List[float] = [0.0 for _ in range(scenario.get_n_subcampaigns())]

        self.current_user_class = 0
        self.current_phase = self.get_current_phase()

        # Collecting additional information
        self.day_breakpoints: List[int] = []
        self.user_count = 0

    def round(self, price) -> (int, bool):
        """
        Price a user of a certain class with a certain price

        :param price: price to be used
        :return: (true if the user buys the product false otherwise, true if the day users are over false otherwise)
        """
        self.user_count += 1
        return self.current_phase.get_crp(class_idx=self.current_user_class, price=price), len(self.sampled_users) == 0

    def next_day(self) -> bool:
        """
        Next day has begin: sample daily users users from the current budget allocation.

        :return true if users has arrived during the next day, false otherwise
        """
        # Check if the day is over: in this case, samples users from the number of visits distribution
        self.current_phase = self.get_current_phase()
        self.sampled_users = []

        n_users_list = self.current_phase.get_all_n_clicks(budget_allocation=self.budget_allocation)
        for i in range(len(n_users_list)):
            min_contexts = self.scenario.get_min_contexts_for_subcampaign(i)
            for min_context in min_contexts:
                self.sampled_users.extend([min_context] * int((n_users_list[i]) / len(min_contexts)))

        np.random.shuffle(self.sampled_users)

        self.daily_users = np.array([self.scenario.get_min_context_to_subcampaign_dict().get(context)
                                     for context in self.sampled_users])

        self.day_t += 1
        self.day_breakpoints.append(self.user_count)

        return len(self.daily_users) > 0

    def next_user(self) -> (Tuple[int, ...], int):
        """
        Removes the current users from the list of the sampled users for the current day and
        returns its features and its class

        :return: current user features, and the class of the user
        """
        current_user_min_context = self.sampled_users.pop()
        self.current_user_class = self.scenario.get_min_context_to_subcampaign_dict().get(current_user_min_context)
        return current_user_min_context, self.current_user_class

    def set_budget_allocation(self, budget_allocation: List[float]) -> None:
        """
        :param budget_allocation: budget allocation for each sub-campaign
        :return: None
        """
        self.budget_allocation = budget_allocation

    def get_day_breakpoints(self) -> List[int]:
        """
        :return: day breakpoints w.r.t. of the number of users arrived in each day
        """
        return self.day_breakpoints

    def get_daily_visits_per_sub_campaign(self) -> List[int]:
        """
        :return: daily visits for each sub-campaign
        """
        return [len(np.where(self.daily_users == i)[0]) for i in range(self.scenario.get_n_subcampaigns())]
