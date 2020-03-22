from typing import List

from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.GP_Regressor import GP_Regressor
from bandit.discrete.CombinatorialBandit import CombinatorialBandit


class CombinatorialGPBandit(CombinatorialBandit):
    """
    Combinatorial bandit that uses GP regressors in order to estimate the number of clicks of each the sub-campaign
    """

    def __init__(self, campaign):
        super().__init__(campaign)
        self.model_list = [GP_Regressor() for _ in range(self.campaign.get_n_sub_campaigns())]

    def pull_arm(self) -> List[float]:
        # Find the best allocation of budgets by optimizing the combinatorial problem of the campaign
        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
        return best_budgets

    def update(self, pulled_arm: List[float], observed_reward: List[float]):
        self.t += 1
        self.update_observations(pulled_arm, observed_reward)
        for sub_index, model in enumerate(self.model_list):
            # Update GP model
            model.update_model(pulled_arm[sub_index], observed_reward[sub_index])

            # Update estimations of the values of the sub-campaigns
            sub_campaign_values = self.model_list[sub_index].sample_gp_distribution(list(self.campaign.get_budgets()))
            self.campaign.set_sub_campaign(sub_index, sub_campaign_values)
