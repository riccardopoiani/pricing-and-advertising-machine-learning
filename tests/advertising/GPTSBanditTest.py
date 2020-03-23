import unittest
import matplotlib.pyplot as plt
import numpy as np

from bandit.discrete.TSBanditGP import TSBanditGP
from environments.AdEnvironment import AdEnvironment


class GP_TSBanditTestCase(unittest.TestCase):
    def test_bandit_process(self):

        def fun(x: np.ndarray) -> np.ndarray:
            return np.array(np.sqrt(x) + 2, dtype=np.int)

        rewards_per_experiment = []

        for i in range(10):
            arms = np.linspace(0, 100, 5, dtype=np.float)
            env = AdEnvironment(fun, arms, 1)
            learner = TSBanditGP(list(arms), 5, normalized=True)

            for j in range(100):
                arm_to_pull = learner.pull_arm()
                reward = env.round(arm_to_pull)
                learner.update(arm_to_pull, reward)

            rewards_per_experiment.append(learner.collected_rewards)

        regrets = []
        for i in range(len(rewards_per_experiment)):
            regrets.append([12 - x for x in rewards_per_experiment[i]])
        avg_regrets = np.mean(regrets, axis=0)
        cum_regrets = np.cumsum(avg_regrets)
        plt.plot(avg_regrets, 'r')
        plt.show()


if __name__ == '__main__':
    unittest.main()
