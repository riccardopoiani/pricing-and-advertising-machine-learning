from abc import ABC, abstractmethod
from typing import List

import numpy as np


class DiscreteRegressor(ABC):
    """
    Abstract class for discrete regressor in bandit scenarios
    """

    def __init__(self, arms: List[float], init_std_dev: float):
        self.arms: List[float] = arms

        self.init_std_dev = init_std_dev
        self.sigmas: np.ndarray = np.ones(len(self.arms)) * self.init_std_dev
        self.means: np.ndarray = np.zeros(len(self.arms))

    def reset_parameters(self) -> None:
        """
        Reset means and standard deviation to their initial values

        :return: None
        """
        self.sigmas: np.ndarray = np.ones(len(self.arms)) * self.init_std_dev
        self.means: np.ndarray = np.zeros(len(self.arms))

    @abstractmethod
    def fit_model(self, collected_rewards: np.array, pulled_arm_history: np.array) -> None:
        """
        Update the model with the given data.
        If the array of given data is empty, the parameters are set to their initial values

        :param pulled_arm_history: array containing the index of the pulled arms at each time-step
        :param collected_rewards: collected rewards at each time-step
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
