import numpy as np

from typing import List

from bandit.discrete.DiscreteBandit import DiscreteBandit


class GIROBernoulliBandit(DiscreteBandit):
    """
    Garbage-in reward-out bandit from:
    "Garbage-in reward-out: bootstrapping exploration in multi-armed bandits" - Kveton et. al [2019]
    """

    def __init__(self, n_arms: int, a: float):
        super().__init__(n_arms=n_arms)
        self.real_reward_count = np.zeros(n_arms)
        self.a = a
        self.pseudo_reward_per_arm: List = [[] for _ in range(n_arms)]

    def _sample_with_replacement(self, arm: int) -> np.array:
        """
        Sample with replacement from the history of the collected rewards of a given arm.
        The replacement is carried out in order to recreate an array with the same size of the current
        available history for the rewards

        :param arm: arm considered from the replacement
        :return: perturbated history obtained with sample with replacement
        """
        arm_rewards = self.rewards_per_arm[arm]
        arm_pseudo_reward = self.pseudo_reward_per_arm[arm]
        history = arm_rewards + arm_pseudo_reward
        return np.random.choice(history, size=len(history), replace=True)

    def pull_arm(self):
        """
        Retrieve the arm to be pulled according to the current statistics.
        In this case exploration is guaranteed via means of sampling with replacement from an history that uses
        both observed rewards and "fake noisy rewards" inserted when the arms are pulled.

        :return: index of the pulled arm
        """
        # Estimate arm values
        estimated_arm_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.real_reward_count[arm] > 0:
                new_arm_history: np.array = self._sample_with_replacement(arm=arm)
                estimated_arm_values[arm] = new_arm_history.mean()
            else:
                estimated_arm_values[arm] = np.infty

        # Select best arm
        max_value = estimated_arm_values.max()
        idxes = np.argwhere(estimated_arm_values == max_value).reshape(-1)
        pulled_arm = np.random.choice(idxes)
        return pulled_arm

    def update(self, pulled_arm, reward):
        """
        Update statistics regarding the arms.
        In particular, only the pulled arm is updated. The peculiarity is that, for the pulled arm, "fake noisy"
        rewards are added to the history of rewards of that arm, according to the parameter a given in the constructor.

        :param pulled_arm: arm that has been pulled
        :param reward: observed reward
        :return: none
        """
        self.t += 1
        self.real_reward_count[pulled_arm] += 1
        self.update_observations(pulled_arm=pulled_arm, reward=reward)

        n_total_pseudo_rewards = int(self.real_reward_count[pulled_arm] * self.a)
        if n_total_pseudo_rewards > len(self.pseudo_reward_per_arm[pulled_arm]) // 2:
            n_new_psuedo_rewards = n_total_pseudo_rewards - (len(self.pseudo_reward_per_arm[pulled_arm]) // 2)

            ones_list = [1 for _ in range(n_new_psuedo_rewards)]
            zeros_list = [0 for _ in range(n_new_psuedo_rewards)]
            self.pseudo_reward_per_arm[pulled_arm].extend(ones_list)
            self.pseudo_reward_per_arm[pulled_arm].extend(zeros_list)
