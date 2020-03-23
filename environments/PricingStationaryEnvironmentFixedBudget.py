import numpy as np
from typing import List

from environments.Environment import Environment
from utils.stats.StochasticFunction import IStochasticFunction


class PricingStationaryEnvironmentFixedBudget(Environment):
    """
    Environment for the pricing scenario
    """

    def __init__(self, conversion_rate_probabilities: List[IStochasticFunction],
                 number_of_visit: List[IStochasticFunction], fixed_budget):
        """
        Create a pricing stationary environment with a fixed budget

        :param conversion_rate_probabilities: distributions of the conversions rate probabilities given the price.
        One element for each class of users
        :param number_of_visit: distributions of the number of visits, given the budget. One element for each
        class of users
        :param fixed_budget: fixed budget to be used in order to simulate the number of visits
        """
        super().__init__(conversion_rate_probabilities=conversion_rate_probabilities,
                         number_of_visit=number_of_visit)
        self.sampled_users: List = []
        self.fixed_budget = fixed_budget
        self.current_user_class = 0
        self.number_of_days = 1

    def round(self, price):
        """
        Simulate if a user buys or not the product

        :param price: price used for the user
        :return: 1 if the users has bought the item, 0 otherwise
        """
        return self.crp_list[self.current_user_class].draw_sample(x=price)

    def get_observation(self) -> int:
        """
        Determine what is the class of the user that is arrived on the website.

        :return: index of the class of the users that is arrived on the website
        """
        # Check if the day is over: in this case, samples users from the number of visits distribution
        while len(self.sampled_users) == 0:
            for i, user_class in enumerate(self.number_of_visit_list):
                n_users = int(user_class.draw_sample(x=self.fixed_budget))
                self.sampled_users.extend([i] * n_users)
            np.random.shuffle(self.sampled_users)

        self.number_of_days += 1
        self.current_user_class = self.sampled_users.pop()
        return self.current_user_class

    def get_number_of_days(self):
        return self.number_of_days
