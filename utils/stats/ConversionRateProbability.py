from abc import ABC, abstractmethod


class ConversionRateProbability(ABC):
    """
    Abstract class representing a conversion rate probability
    """

    def __init__(self):
        super().__init__()

    """
    Draw a sample from the conversion rate probability function.
    
    :param price: price at which the sample is drawn
    :return sample from the conversion rate probability
    """
    @abstractmethod
    def draw_sample(self, price) -> float:
        pass
