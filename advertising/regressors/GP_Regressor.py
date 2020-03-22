from typing import List

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF


class GP_Regressor(object):
    """
    1D-input Gaussian Process Regressor in order to estimate a function
    """

    def __init__(self, alpha: float = 10, n_restarts_optimizer: int = 5):
        self.input_data = []
        self.label_data = []

        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, normalize_y=True,
                                           n_restarts_optimizer=n_restarts_optimizer)

    def update_model(self, x: float, label: float):
        """
        Update the Gaussian Process model in an online fashion

        :param x: scalar input value
        :param label: scalar output value
        :return:
        """
        self.input_data.append(x)
        self.label_data.append(label)

        self.gp.fit(np.atleast_2d(self.input_data).T, self.label_data)

    def sample_gp_distribution(self, inputs: List[float]):
        """
        Samples the GP distribution on the given 'inputs'

        :param inputs: list of scalar values to sample
        :return: list of scalar values, i.e. samples of the gaussian distribution based on the means and sigmas of
        the GP prediction
        """
        # TODO: it is possible to sample in another way "mean" just like in the paper
        means, sigmas = self.gp.predict(np.atleast_2d(inputs).T, return_std=True)
        sigmas = np.maximum(sigmas, 1e-2)  # avoid negative numbers
        return np.random.normal(means, sigmas)
