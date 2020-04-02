import numpy as np
from bandit.discrete.DiscreteBandit import DiscreteBandit


class UCB1MBandit(DiscreteBandit):
    """
    Class representing a UCB1-M bandit: exploits the monotonicity assumption of the customer's demand curve
    (the larger the price, the smaller the conversion probability)
    Found at https://home.deib.polimi.it/trovo/01papers/trovo2018improving_a.pdf
    """

    def __init__(self, n_arms: int, arm_values: np.array):
        super().__init__(n_arms)
        self.arm_values: np.array = arm_values
        self.round_per_arm: np.array = np.zeros(n_arms)
        self.expected_bernoulli: np.array = np.zeros(n_arms)
        self.upper_bound: np.array = np.ones(n_arms)
        self.max_value: float = np.max(arm_values)

    def pull_arm(self) -> int:
        """
        Decide which arm to pull:
        - every arm needs to be pulled at least once (randomly)
        - if every arm has been pulled at least once, the arm with the highest (upper_bound * price) will be pulled
        (ties are randomly broken)

        :return pulled_arm: the index of the pulled arm
        """
        if self.t < self.n_arms:
            return np.random.choice(np.argwhere(self.round_per_arm == 0).reshape(-1))

        upper_bound_with_price = self.upper_bound * self.arm_values
        idxes = np.argwhere(upper_bound_with_price == upper_bound_with_price.max()).reshape(-1)
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
        - the expected bernoulli of the pulled arm
        - the upper bound of all the arms

        :param pulled_arm: arm that has been pulled
        :param reward: reward obtained pulling pulled_arm
        :return: none
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)

        reward = reward / self.max_value

        # update the number of times the arm has been pulled
        self.round_per_arm[pulled_arm] += 1

        # update the expected bernoulli of the pulled arm
        if reward != 0:
            self.expected_bernoulli[pulled_arm] = (self.expected_bernoulli[pulled_arm] *
                                                   (self.round_per_arm[pulled_arm]-1) +
                                                   (reward/reward)) / self.round_per_arm[pulled_arm]
        else:
            self.expected_bernoulli[pulled_arm] = (self.expected_bernoulli[pulled_arm] *
                                                   (self.round_per_arm[pulled_arm] - 1)) / self.round_per_arm[pulled_arm]

        # for each arm a, update its upper confidence bound
        for a in range(0, self.n_arms):
            bound_list = []
            # for each possible starting_arm, compute one possible bound for arm a
            for starting_arm in range(0, a+1):
                # sum the number of times the arms from starting_arm to a have been pulled
                round_per_previous_arms = self.round_per_arm[starting_arm:a+1].sum()

                non_normalized_expected_bernoulli = 0
                for arm in range(starting_arm, a+1):
                    non_normalized_expected_bernoulli += self.round_per_arm[arm] * self.expected_bernoulli[arm]

                expected_bernoulli_per_previous_arm = non_normalized_expected_bernoulli / round_per_previous_arms
                bound_list.append(
                    expected_bernoulli_per_previous_arm + np.sqrt(((4 * np.log(self.t)) + np.log(a+1)) /
                                                               (2 * round_per_previous_arms)))
            # minimize the list of possible bounds by the starting_arm
            # and assign the minimized term to the upper bound of arm a
            self.upper_bound[a] = min(bound_list)

    def get_optimal_arm(self) -> int:
        expected_rewards = self.expected_bernoulli*self.arm_values
        return int(np.argmax(expected_rewards))
