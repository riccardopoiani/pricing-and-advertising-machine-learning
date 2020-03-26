import unittest
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from bandit.discrete.TSBanditGP import TSBanditGP




class GP_TSBanditTestCase(unittest.TestCase):
    def test_bandit_process(self):

        def fun(x: np.ndarray) -> np.ndarray:
            return np.minimum(np.array(40*x, dtype=np.int), 2000)

        rewards_per_experiment = []

        for i in range(1):
            arms = np.linspace(0, 100, 11, dtype=np.float)
            learner = TSBanditGP(list(arms), 0, init_std_dev=1e3, alpha=5, normalized=True)

            for j in range(150):
                arm_to_pull = learner.pull_arm()
                reward = fun(arms[arm_to_pull])+np.random.normal(0, 1)
                learner.update(arm_to_pull, reward)

                means = learner.gp_regressor.means
                sigmas = learner.gp_regressor.sigmas

                if j > 100:
                    plt.figure(j - 100)

                    plt.plot(arms, fun(arms), 'r:', label=r'$fun(x)$')
                    plt.plot(arms[learner.pulled_arm_list], learner.collected_rewards, 'ro', label=u'Observed Clicks')
                    plt.plot(arms, means, 'b--', label=u'Predicted Clicks')
                    plt.fill(np.concatenate([arms, arms[::-1]]),
                             np.concatenate([means - 1.96*sigmas, (means + 1.96*sigmas)[::-1]]),
                             alpha=.5, fc='b', ec='None', label='95% conf interval')
                    plt.xlabel('$x$')
                    plt.ylabel('$fun(x)$')
                    plt.legend(loc='lower right')
                    plt.show()

            rewards_per_experiment.append(learner.collected_rewards)

        regrets = []
        for i in range(len(rewards_per_experiment)):
            regrets.append([12 - x for x in rewards_per_experiment[i]])
        avg_regrets = np.mean(regrets, axis=0)
        cum_regrets = np.cumsum(avg_regrets)
        plt.plot(avg_regrets, 'g')
        plt.show()

        plt.plot(cum_regrets, 'r')
        plt.show()


if __name__ == '__main__':
    unittest.main()
