from typing import List

from advertising.regressors import DiscreteRegressor
from advertising.regressors.DiscreteGaussianRegressor import DiscreteGaussianRegressor
from bandit.discrete.CombinatorialBandit import CombinatorialBandit


class CombinatorialGaussianBandit(CombinatorialBandit):
    """
    Combinatorial bandit that uses gaussian regressors in order to estimate the number of clicks of each
    the sub-campaign
    """

    def __init__(self, campaign, init_std_dev: float = 1e3):
        super().__init__(campaign)
        self.model_list: List[DiscreteRegressor] = [DiscreteGaussianRegressor(list(self.campaign.get_budgets()),
                                                                              init_std_dev)
                                                    for _ in range(self.campaign.get_n_sub_campaigns())]
