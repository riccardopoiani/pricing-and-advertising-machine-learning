from abc import ABC, abstractmethod


class Distribution(ABC):
    """
    Abstract class representing a distribution from which it is possible to draw samples
    """

    def __init__(self):
        super().__init__()

    """
    Draw a sample from a distribution

    :return sample from the distribution
    """
    @abstractmethod
    def draw_sample(self) -> float:
        pass
