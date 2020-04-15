from abc import ABC, abstractmethod
from typing import Callable, List


class IStochasticFunction(ABC):
    """
    Abstract class representing a stochastic function
    """

    def __init__(self):
        super().__init__()

    """
    Draw a sample from the stochastic function.
    
    :param x: point at which the sample is drawn
    :return sample from stochastic function
    """

    @abstractmethod
    def draw_sample(self, x) -> float:
        pass


class BoundedLambdaStochasticFunction(IStochasticFunction):
    """
    Stochastic function implemented by means of a callable object that
    requires one float argument (e.g. the price in the case of a conversion rate probability)
    """

    def __init__(self, f: Callable, min_value: float = None, max_value: float = None):
        """
        Create a stochastic function using a callable object

        :param f: callable object that requires a float argument (e.g. the price for a conversion rate probability)
        :param min_value: optional minimum price
        :param max_value: optional maximum price
        """
        super().__init__()
        self.f = f
        self.min_value = min_value
        self.max_value = max_value

    def draw_sample(self, x: float) -> float:
        if self.min_value is not None:
            assert x >= self.min_value
        if self.max_value is not None:
            assert x <= self.max_value

        return self.f(x)


class MultipliedStochasticFunction(IStochasticFunction):
    """
    It builds a new function multiplying the drawn sample by the values with which the function is called
    shifted by some specified quantity
    """

    def __init__(self, f: IStochasticFunction, shift):
        super(IStochasticFunction).__init__()
        self.f = f
        self.shift = shift

    def draw_sample(self, x) -> float:
        return self.f.draw_sample(x) * (x + self.shift)

    def get_minus_lambda(self) -> Callable:
        return lambda x: -self.draw_sample(x)


class AggregatedFunction(IStochasticFunction):
    """
    Aggregated function: it uses different functions and weight the results of each sample
    to produce a sample
    """

    def __init__(self, f_list: List[IStochasticFunction], weights):
        super(IStochasticFunction).__init__()

        assert len(f_list) == len(weights)

        self.f_list: List[IStochasticFunction] = f_list
        self.weights = weights

    def draw_sample(self, x) -> float:
        res = 0
        for i, f in enumerate(self.f_list):
            res += f.draw_sample(x) * self.weights[i]
        return res

    def get_minus_lambda(self) -> Callable:
        return lambda x: -self.draw_sample(x)
