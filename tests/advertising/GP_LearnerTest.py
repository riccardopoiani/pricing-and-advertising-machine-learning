import unittest
import matplotlib.pyplot as plt
import numpy as np

from bandit.discrete.GP_TSBandit import GP_TSBandit
from environments.AdEnvironment import AdEnvironment


class MyTestCase(unittest.TestCase):
    def test_something(self):

        def fun(x: np.ndarray) -> np.ndarray:
            return np.array(np.sqrt(x) + 2, dtype=np.int)

        rewards_per_experiment = []

        for i in range(5):
            arms = np.linspace(0, 100, 5)
            env = AdEnvironment(fun, arms, 2)
            learner = GP_TSBandit(list(arms), 5)

            for j in range(60):
                arm_to_pull = learner.pull_arm()
                reward = env.round(arm_to_pull)
                learner.update(arm_to_pull, reward)

            rewards_per_experiment.append(learner.collected_rewards)

        avg_regrets = np.mean([12*60 - x for x in rewards_per_experiment], axis=0)
        cum_regrets = np.cumsum(avg_regrets)
        plt.plot(cum_regrets, 'r')
        plt.show()


if __name__ == '__main__':
    unittest.main()
