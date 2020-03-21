import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

from bandit.discrete.DiscreteBandit import DiscreteBandit


class GPBandit(DiscreteBandit):

    def __init__(self, arms: np.ndarray, init_std_dev: float, alpha: float = 10, n_restarts_optimizer: int = 5):
        super().__init__(len(arms))

        self.arms = arms
        self.means = np.zeros(self.n_arms)
        self.sigmas = np.ones(self.n_arms) * init_std_dev
        self.pulled_arms = []
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                           n_restarts_optimizer=n_restarts_optimizer)

    def update_observations(self, pulled_arm: int, reward: float):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)  # avoid negative numbers

    def update(self, pulled_arm: int, reward: float):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def sample(self):
        # TODO: it is possible to sample in another way "mean" just like in the paper
        return np.random.normal(self.means, self.sigmas)
