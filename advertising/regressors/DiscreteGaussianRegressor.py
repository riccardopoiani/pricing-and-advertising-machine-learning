import numpy as np

from advertising.regressors.DiscreteRegressor import DiscreteRegressor


class DiscreteGaussianRegressor(DiscreteRegressor):
    """
    1D-input discrete Gaussian Regressor in order to estimate a function
    """

    def __init__(self, arms, init_std_dev=1e3):
        super().__init__(arms, init_std_dev)

    def update_model(self, pulled_arm: int, reward: float):

        self.pulled_arm_list.append(pulled_arm)
        self.rewards_per_arm[pulled_arm].append(reward)

        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = len(self.rewards_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples

    def sample_distribution(self):
        return np.random.normal(self.means, self.sigmas)
