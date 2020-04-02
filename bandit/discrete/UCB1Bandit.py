import numpy as np
from bandit.discrete.DiscreteBandit import DiscreteBandit


class UCB1Bandit(DiscreteBandit):
    """
    Class representing a UCB1 bandit
    """

    def __init__(self, n_arms: int, arm_values: np.array):
        super().__init__(n_arms)
        self.round_per_arm: np.array = np.zeros(n_arms)
        self.expected_rewards: np.array = np.zeros(n_arms)
        self.upper_bound: np.array = np.ones(n_arms)
        self.max_value: float = np.max(arm_values)

    def pull_arm(self) -> int:
        """
        Decide which arm to pull:
        - every arm needs to be pulled at least once (randomly)
        - if every arm has been pulled at least once, the arm with the highest upper bound will be pulled
        (ties are randomly broken)

        :return pulled_arm: the index of the pulled arm
        """
        if self.t < self.n_arms:
            return np.random.choice(np.argwhere(self.round_per_arm == 0).reshape(-1))
        idxes = np.argwhere(self.upper_bound == self.upper_bound.max()).reshape(-1)
        pulled_arm = np.random.choice(idxes)
        return pulled_arm

    def update(self, pulled_arm, reward) -> None:
        """
        Update bandit statistics:
        - the reward collected for a given arm from the beginning of the learning process
        - ordered list containing the rewards collected from the beginning of the learning process
        - the list of pulled arms from round 0 to round t
        - the round number t
        - the numpy array containing the number of times each arm has been pulled until the current round t
        - the expected reward of the pulled arm
        - the upper bound of all the arms

        :param pulled_arm: arm that has been pulled
        :param reward: reward obtained pulling pulled_arm
        :return: none
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)

        # Update rewards to deal with random variables not in [0, 1]
        reward = reward / self.max_value

        # update the number of times the arm has been pulled
        self.round_per_arm[pulled_arm] += 1

        # update the expected reward of the pulled arm
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.round_per_arm[pulled_arm]-1) +
                                             reward) / self.round_per_arm[pulled_arm]
        # update upper confidence bound
        self.upper_bound = self.expected_rewards + np.sqrt((2 * np.log(self.t) / self.round_per_arm))

    def get_optimal_arm(self) -> int:
        return int(np.argmax(self.expected_rewards))
