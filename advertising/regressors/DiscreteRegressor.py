from abc import ABC, abstractmethod
from typing import List

import numpy as np


class DiscreteRegressor(ABC):
    """
    Abstract class for discrete regressor in bandit scenarios
    """

    def __init__(self, arms: List[float], init_std_dev: float):
        self.arms: List[float] = arms
        self.pulled_arm_list: List[int] = []
        self.rewards_per_arm: List[List] = [[] for _ in range(len(arms))]

        self.means: np.ndarray = np.zeros(len(self.arms))
        self.sigmas: np.ndarray = np.ones(len(self.arms)) * init_std_dev

    @abstractmethod
    def update_model(self, pulled_arm: int, reward: float) -> None:
        """
        Update the Gaussian Process model in an online fashion

        :param pulled_arm: index of the pulled arm
        :param reward: scalar reward observation value
        :return: None
        """
        pass

    @abstractmethod
    def sample_distribution(self) -> List[float]:
        """
        Samples the distribution of the model on the 'arms'

        :return: list of scalar values, i.e. samples of the distribution based on the means and sigmas
        """
        pass
