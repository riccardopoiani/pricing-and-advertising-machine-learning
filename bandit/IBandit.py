from abc import ABC, abstractmethod


class IBandit(ABC):

    @abstractmethod
    def pull_arm(self):
        pass

    @abstractmethod
    def update(self, pulled_arm, observed_reward):
        pass
