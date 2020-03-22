from typing import List

from utils.stats.BernoulliDistribution import BernoulliDistribution


class BernoulliDiscreteBanditEnv(object):
    """
    Environment for discrete arms whose reward is retrieved from a Bernoulli
    distribution
    """

    def __init__(self, n_arms: int, probabilities: List[BernoulliDistribution]):
        super().__init__()
        self.n_arms: int = n_arms
        self.probabilities: List[BernoulliDistribution] = probabilities

    """
    Round of the environment (e.g. a users has received a certain price from the
    website)
    
    :return: whether the user has purchased or not the item 
    """
    def round(self, pulled_arm: int) -> float:
        reward = self.probabilities[pulled_arm].draw_sample()
        return reward
