from typing import List

from advertising.data_structure.Campaign import Campaign
from advertising.regressors import DiscreteRegressor
from advertising.regressors.DiscreteGPRegressor import DiscreteGPRegressor
from bandit.discrete.CombinatorialBandit import CombinatorialBandit


class CombinatorialGPBandit(CombinatorialBandit):
    """
    Combinatorial bandit that uses GP regressors in order to estimate the number of clicks of each
    the sub-campaign
    """

    def __init__(self, campaign: Campaign, init_std_dev: float = 1e3, alpha: float = 10, n_restarts_optimizer=5):
        super().__init__(campaign)
        self.model_list: List[DiscreteRegressor] = [DiscreteGPRegressor(list(campaign.get_budgets()), init_std_dev,
                                                                        alpha, n_restarts_optimizer)
                                                    for _ in range(self.campaign.get_n_sub_campaigns())]

