import numpy as np

from advertising.regressors.DiscreteRegressor import DiscreteRegressor


class DiscreteGaussianRegressor(DiscreteRegressor):
    """
    1D-input discrete Gaussian Regressor in order to estimate a function
    """

    def __init__(self, arms, init_std_dev=1e3):
        super().__init__(arms, init_std_dev)

    def fit_model(self, collected_rewards: np.array, pulled_arm_history: np.array):
        count_pulled_arm = np.where(pulled_arm_history == pulled_arm_history[-1])[0].size
        self.means[pulled_arm_history[-1]] = (self.means[pulled_arm_history[-1]] * (count_pulled_arm - 1)
                                              + collected_rewards[-1]) / count_pulled_arm

        if count_pulled_arm > 1:
            count_pulled_arm = count_pulled_arm - 1
            self.sigmas[pulled_arm_history[-1]] = (count_pulled_arm / (count_pulled_arm + 1)) * self.sigmas[pulled_arm_history[-1]] \
                                                  + (1 / count_pulled_arm) * ((collected_rewards[-1] - self.means[pulled_arm_history[-1]]) ** 2)

    def sample_distribution(self):
        return np.random.normal(self.means, self.sigmas)
