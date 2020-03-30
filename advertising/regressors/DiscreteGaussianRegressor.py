import numpy as np

from advertising.regressors.DiscreteRegressor import DiscreteRegressor


class DiscreteGaussianRegressor(DiscreteRegressor):
    """
    1D-input discrete Gaussian Regressor in order to estimate a function
    """

    def __init__(self, arms, init_std_dev=1e3):
        super().__init__(arms, init_std_dev)

    def fit_model(self, collected_rewards: np.array, pulled_arm_history: np.array):
        # TODO this can be optimized
        for arm in self.arms:
            arm_index = np.where(pulled_arm_history == arm)
            self.means[arm] = collected_rewards[arm_index].mean()
            self.sigmas[arm] = collected_rewards[arm_index].std()

    def sample_distribution(self):
        return np.random.normal(self.means, self.sigmas)
