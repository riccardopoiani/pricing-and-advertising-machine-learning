import numpy as np
from bandit.discrete.DiscreteBandit import DiscreteBandit


class UCB1MBandit(DiscreteBandit):
    """
    Class representing a UCB1-M bandit
    """

    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.round_per_arm = np.zeros(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        self.upper_bound = np.ones(n_arms)

    def pull_arm(self):
        """
        Decide which arm to pull:
        - every arm needs to be pulled at least once
        - if every arm has been pulled at least once, the arm with the highest upper bound will be pulled
        (ties are randomly broken)

        :return pulled_arm: the array index of the pulled arm
        """
        if self.t < self.n_arms:
            return np.random.choice(np.argwhere(self.round_per_arm == 0).reshape(-1))
        #print(self.upper_bound)
        #print(type(self.upper_bound))
        idxes = np.argwhere(self.upper_bound == self.upper_bound.max()).reshape(-1)
        pulled_arm = np.random.choice(idxes)
        return pulled_arm

    def update(self, pulled_arm, reward):
        """
        Update bandit statistics:
        - the reward collected for a given arm from the beginning of the learning process
        - ordered list containing the rewards collected from the beginning of the learning process
        - the round number t
        - the numpy array containing the number of times each arm has been pulled until the current round t
        - the expected reward of the pulled arm
        - the upper bound of the pulled arm

        :param pulled_arm: arm that has been pulled
        :param reward: reward obtained pulling pulled_arm
        :return: none
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        # update the number of times the arm has been pulled
        self.round_per_arm[pulled_arm] += 1
        # update the expected reward of the pulled arm
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.round_per_arm[pulled_arm] - 1) +
                                             reward) / self.round_per_arm[pulled_arm]
        # update upper confidence bounds
        for a in range(0, self.n_arms):
            bound_list = []
            for starting_arm in range(0, a+1):
                round_per_previous_arms = sum(self.round_per_arm[starting_arm:a+1])

                non_normalized_expected_reward = 0
                for arm in range(starting_arm, a+1):
                    non_normalized_expected_reward += self.round_per_arm[arm] * self.expected_rewards[arm]

                expected_reward_per_previous_arm = non_normalized_expected_reward / round_per_previous_arms
                bound_list.append(
                    expected_reward_per_previous_arm + np.sqrt(((4 * np.log(self.t)) + np.log(a+1)) /
                                                               (2 * round_per_previous_arms)))
            self.upper_bound[a] = min(bound_list)
