import numpy as np


class DiscreteBandit(object):
    """
    General class representing a bandit that uses a discrete number of arms
    """

    def __init__(self, n_arms):
        self.t = np.random.binomial(0, 0)
        self.n_arms = n_arms
        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = []

    def update_observations(self, pulled_arm, reward):
        """
        Update bandit statistics:
        - the reward collected for a given arm from the beginning of the learning process
        - ordered list containing the rewards collected from the beginning of the learning process

        :param pulled_arm: arm that has been pulled
        :param reward: reward obtained pulling pulled_arm
        :return: none
        """
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards.append(reward)
