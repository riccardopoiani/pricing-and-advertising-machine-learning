import numpy as np

from bandit.discrete.TSBandit import TSBandit


class TSBanditBernoulli(TSBandit):
    """
    Class representing a bandit that operates with Bernoulli distributions
    """

    def __init__(self, n_arms):
        super(TSBandit, self).__init__(n_arms)
        self.beta_distribution = np.ones((n_arms, 2))

    def sample_beta_distribution(self) -> np.array:
        prior_sampling = np.random.beta(a=self.beta_distribution[:, 0],
                                        b=self.beta_distribution[:, 1])
        return prior_sampling

    def update_beta_distribution(self, pulled_arm, reward):
        self.beta_distribution[pulled_arm, 0] += reward
        self.beta_distribution[pulled_arm, 1] += (1.0 - reward)

