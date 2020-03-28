import numpy as np

from bandit.discrete.DiscreteBandit import DiscreteBandit


class TSBanditRescaledBernoulli(DiscreteBandit):
    """
    Class representing a Thompson sampling discrete learner
    """

    def __init__(self, n_arms: int, arm_values: np.array):
        super().__init__(n_arms=n_arms)
        self.beta_distribution: np.array = np.ones((n_arms, 2))
        self.arm_values: np.array = arm_values

    def pull_arm(self) -> int:
        """
        Retrieve the index of the arm to be pulled by the bandit according to the current
        statistics

        :return: index of the arm to be pulled
        """
        idx = np.argmax(self.sample_beta_distribution() * self.arm_values)
        # TODO fix warning
        return idx

    def update(self, pulled_arm, reward) -> None:
        """
        Update the bandit statistics with an observation tuple (pulled_arm, observed reward)

        :param pulled_arm: arm that has been pulled
        :param reward: reward that has been observed
        :return: none
        """
        self.t += 1
        self.update_observations(pulled_arm=pulled_arm, reward=reward)
        self.update_beta_distribution(pulled_arm=pulled_arm, reward=reward)

    def sample_beta_distribution(self) -> np.array:
        """
        Draw a sample from the prior distribution of each arm of the bandit

        :return: array containing in the i-th position the sample obtained from the
        prior of the i-th arm of the bandit
        """

        prior_sampling = np.random.beta(a=self.beta_distribution[:, 0],
                                        b=self.beta_distribution[:, 1])
        return prior_sampling

    def update_beta_distribution(self, pulled_arm, reward) -> None:
        """
        Update the beta distribution of the pulled arm after having observed a reward
        related to it

        :param pulled_arm: index of the arm that has been pulled
        :param reward: observed reward of pulled_arm
        :return: none
        """
        if reward != 0:
            reward = 1
        self.beta_distribution[pulled_arm, 0] += reward
        self.beta_distribution[pulled_arm, 1] += (1.0 - reward)
