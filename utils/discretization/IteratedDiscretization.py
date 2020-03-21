import numpy as np

from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod
from typing import List, Callable
from scipy.optimize import minimize_scalar


class IteratedDiscretization(ABC):

    def __init__(self, reset_interval: int,
                 min_x, max_x, initial_interval_size: float):
        super().__init__()

        # General information
        self.interval_size = initial_interval_size

        # Hyper-parameter
        self.reset_interval = reset_interval

        # Dataset information
        self.arm_history = []
        self.reward_history = []

        # Variable boundary
        self.min_x = min_x
        self.max_x = max_x

    @abstractmethod
    def fit_model(self) -> Callable:
        """
        Fit a certain model with the available data and return a callable function that returns
        the model prediction in the given point

        :return: callable function that returns model prediction in the given point
        """
        pass

    def discretize(self, model, n_arms) -> np.array:
        """
        Discretize uniformly the new obtained interval.
        In particular, the minimum of the given model is found and used a center for the new interval.
        Note that the model is minimized, thus proi

        :param model: model representing the estimated function for each point
        :param n_arms: number of new arms to be obtained
        :return: new arms after the new interval using the given model is found
        """
        opt_result = minimize_scalar(fun=model, bounds=(self.min_x, self.max_x), method="bounded")

        if not opt_result['success']:
            print("Warning: function optimization failed")

        new_central_arm = opt_result['x']

        self.interval_size /= 2

        k1 = new_central_arm - (self.interval_size / 2)
        k2 = new_central_arm + (self.interval_size / 2)
        k1 = np.max((k1, self.min_x))
        k2 = np.min((k2, self.max_x))
        self.interval_size = np.abs(k1 - k2)

        return np.linspace(start=k1, stop=k2, num=n_arms)

    def check_update_condition(self, time: int) -> bool:
        """
        Check if new arms should be found or not according to the specified reset interval

        :param time: current elapsed time from the last discretization
        :return: true if the time elapsed is greater or equal than the reset interval, false otherwise
        """
        if time >= self.reset_interval:
            return True
        return False

    def update_dataset(self, interval_arm_history: List, interval_reward_history: List):
        """
        Update the dataset of collected rewards and pulled arms.
        Note that the pulled arms must be expressed

        :param interval_arm_history:
        :param interval_reward_history:
        :return: none
        """
        self.arm_history.extend(interval_arm_history)
        self.reward_history.extend(interval_reward_history)


class LinearModel(IteratedDiscretization):
    """
    Linear regression model fitted on the available data, which are shuffled
    """

    def fit_model(self) -> Callable:
        n_samples = len(self.reward_history)
        shuffled_idx = np.arange(n_samples)
        np.random.shuffle(shuffled_idx)

        x = np.array(self.arm_history)
        y = np.array(self.reward_history)
        x = x[shuffled_idx]
        y = y[shuffled_idx]

        x = x.reshape(-1, 1)

        reg = LinearRegression().fit(x, y)

        return lambda price: -reg.predict(np.array(price).reshape(-1, 1))