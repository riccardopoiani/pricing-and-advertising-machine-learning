from typing import List

import numpy as np
from bandit.discrete.DiscreteBandit import DiscreteBandit


class EXP3Bandit(DiscreteBandit):
    """
    Class representing a Exp3 bandit
    Found at https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
    """

    def __init__(self, n_arms: int, gamma: float, arm_values: np.array):
        super().__init__(n_arms)

        # Hyper-parameter gamma belonging to [0,1] which tunes the desire to pick an action uniformly at random
        self.gamma: float = gamma

        # Additional data structure
        self.arm_values: np.array = arm_values
        self.max_value: float = np.max(self.arm_values)
        self.weight_per_arm: np.array = np.ones(n_arms)
        self.distribution_per_arm: List = [1 for _ in range(n_arms)]
        self.expected_rewards: np.array = np.zeros(n_arms)

    def pull_arm(self) -> int:
        """
        Decide which arm to pull:
        - set the distributions of the arms
        - draw the next arm randomly according to the distributions of the arms

        :return pulled_arm: the index of the pulled arm
        """
        self.distribution_per_arm: List = self._get_arm_probabilities()
        idx = np.random.choice(a=self.n_arms, size=1, p=self.distribution_per_arm)
        return idx[0]

    def _get_arm_probabilities(self) -> List:
        """
        :return: arm probabilities to be used for choosing the arm
        """
        weight_sum = float(sum(self.weight_per_arm))
        return [(1.0 - self.gamma) * (w / weight_sum) + (self.gamma / self.n_arms) for w in self.weight_per_arm]

    def update(self, pulled_arm, reward) -> None:
        """
        Update bandit statistics:
        - the reward collected for a given arm from the beginning of the learning process
        - ordered list containing the rewards collected from the beginning of the learning process
        - the list of pulled arms from round 0 to round t
        - the round number t
        - the expected reward of the pulled arm
        - the weights of the pulled arm

        :param pulled_arm: arm that has been pulled
        :param reward: reward obtained pulling pulled_arm
        :return: none
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        reward = reward / self.max_value

        # update the expected reward of the pulled arm
        self.expected_rewards[pulled_arm] = reward / self.distribution_per_arm[pulled_arm]

        # update the weight of the pulled arm
        self.weight_per_arm[pulled_arm] *= np.math.exp(self.expected_rewards[pulled_arm] * self.gamma / self.n_arms)
