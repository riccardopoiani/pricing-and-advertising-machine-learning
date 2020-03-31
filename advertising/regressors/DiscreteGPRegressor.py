import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C

from advertising.regressors.DiscreteRegressor import DiscreteRegressor


class DiscreteGPRegressor(DiscreteRegressor):
    """
    1D-input Gaussian Process Regressor in order to estimate a function
    """

    def __init__(self, arms, init_std_dev=1e3, alpha: float = 10, n_restarts_optimizer: int = 10,
                 normalized: bool = True):
        if normalized:
            arms = arms / np.max(arms)
        super().__init__(arms, init_std_dev)

        self.kernel: Product = C(1.0, (1e-8, 1e8)) * RBF(1.0, (1e-8, 1e8))
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer

        self.gp: GaussianProcessRegressor = GaussianProcessRegressor(kernel=self.kernel, alpha=self.alpha ** 2,
                                                                     normalize_y=True,
                                                                     n_restarts_optimizer=self.n_restarts_optimizer)

    def fit_model(self, collected_rewards: np.array, pulled_arm_history: np.array):
        if len(collected_rewards) == 0 or len(pulled_arm_history):
            self.reset_parameters()
        else:
            x = np.atleast_2d(np.array(self.arms)[pulled_arm_history]).T

            self.gp.fit(x, collected_rewards)
            self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
            self.sigmas = np.maximum(self.sigmas, 1e-2)  # avoid negative numbers

    def sample_distribution(self):
        return np.random.normal(self.means, self.sigmas)
