from typing import List
from abc import ABC

from bandit.IBandit import IBandit


class DiscreteBandit(IBandit, ABC):
    """
    General class representing a bandit that uses a discrete number of arms
    """

    def __init__(self, n_arms):
        self.n_arms: int = n_arms
        self.t: int = 0
        self.rewards_per_arm: List = [[] for _ in range(n_arms)]
        self.collected_rewards: List = []
        self.pulled_arm_list: List = []

    def update_observations(self, pulled_arm, reward):
        """
        Update bandit statistics:
        - the reward collected for a given arm from the beginning of the learning process
        - ordered list containing the rewards collected from the beginning of the learning process
        - the list of pulled arms from round 0 to round t

        :param pulled_arm: arm that has been pulled
        :param reward: reward obtained pulling pulled_arm
        :return: none
        """
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards.append(reward)
        self.pulled_arm_list.append(pulled_arm)
