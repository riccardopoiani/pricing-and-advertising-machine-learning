import numpy as np
from utils.stats.Distribution import Distribution


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution with parameter theta
    """

    def __init__(self, theta):
        super(Distribution).__init__()
        self.theta = theta

    def draw_sample(self) -> float:
        return np.random.binomial(1, self.theta)