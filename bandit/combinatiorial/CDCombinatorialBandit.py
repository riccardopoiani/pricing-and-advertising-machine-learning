import numpy as np

from typing import List

from advertising.data_structure import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit

# TODO
#   - find good hyperparameters
#   - find out if, at each change detection, it is better to reset a single sub-campaign or to reset them all
class CDCombinatorialBandit(CombinatorialBandit):
    """
    Change Detection combinatorial bandit
    """

    def __init__(self, campaign: Campaign, model_list: List[DiscreteRegressor], n_arms: int, gamma: float,
                 cd_threshold: float, sw_size: int):
        assert sw_size % 2 == 0

        super().__init__(campaign=campaign, model_list=model_list)
        self.n_arms = n_arms
        self.gamma = gamma
        self.cd_threshold = cd_threshold
        self.sw_size = sw_size
        self.last_detection_time = np.zeros(shape=campaign.get_n_sub_campaigns(), dtype=np.int)

        self.count_arm_after_detection = np.zeros(shape=(campaign.get_n_sub_campaigns(), n_arms))
        self.last_sw_arm_rewards = np.zeros(shape=(campaign.get_n_sub_campaigns(), n_arms, sw_size))

    def pull_arm(self) -> List[int]:
        """
        Find the best allocation of budgets by optimizing the combinatorial problem of the campaign and then return
        the indices of the best budgets.
        The combinatorial problem is optimized given estimates provided only by the last data (amount specified
        by the sliding window)

        :return: the indices of the best budgets given the actual campaign
        """
        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
        return [np.where(self.campaign.get_budgets() == budget)[0][0] for budget in best_budgets]

    def update_observations(self, pulled_arm: List[int], reward: List[float]) -> None:
        """
        Update the combinatorial bandit statistics:
         - ordered list containing the rewards collected since the beginning
         - ordered list of the superarm pulled since the beginning

        :param pulled_arm: the superarm that has been pulled (list of indices)
        :param reward: reward obtained pulling pulled_superarm
        :return: None
        """
        self.collected_rewards.append(sum(reward))
        self.pulled_superarm_list.append(pulled_arm)

        for i in range(self.campaign.get_n_sub_campaigns()):
            self.pulled_arm_sub_campaign[i].append(pulled_arm[i])
            self.collected_rewards_sub_campaign[i].append(reward[i])

        for i in range(self.campaign.get_n_sub_campaigns()):
            self.last_sw_arm_rewards[i][pulled_arm[i]] = np.roll(self.last_sw_arm_rewards[i][pulled_arm[i]], -1)
            self.last_sw_arm_rewards[i][pulled_arm[i]][-1] = reward[i]
            self.count_arm_after_detection[i][pulled_arm[i]] += 1

    def update(self, pulled_arm: List[int], reward: List[float]) -> None:
        """
        Update observations and models of the sub-campaign

        :param pulled_arm: list of indices of the pulled arms (i.e. superarm pulled)
        :param reward: list of observed reward for each pulled arm
        :return: None
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)

        campaign_mask = self._change_detection()
        self.last_detection_time[campaign_mask] = self.t - 1
        self.count_arm_after_detection[campaign_mask] = 0
        self.last_sw_arm_rewards[campaign_mask] = 0

        for i, model in enumerate(self.model_list):
            if np.random.binomial(n=1, p=1 - self.gamma):
                model.fit_model(collected_rewards=self.collected_rewards_sub_campaign[i][self.last_detection_time[i]:
                                                                                         self.t],
                                pulled_arm_history=self.pulled_arm_sub_campaign[i][self.last_detection_time[i]:self.t])
            else:
                model.fit_model(collected_rewards=[],
                                pulled_arm_history=[])

            # Update estimations of the values of the sub-campaigns
            sub_campaign_values = self.model_list[i].sample_distribution()
            self.campaign.set_sub_campaign(i, sub_campaign_values)

    def _change_detection(self) -> np.array:
        """
        Change point detection mechanisms, as described in "Nearly optimal adaptive procedure with change
        detection for piece-wise stationary bandit" [Kveton et. al]

        :return: array containing a boolean for each sub-campaign if it should be re-set or not
        """
        arm_mask = self.count_arm_after_detection >= self.sw_size
        campaign_mask = np.zeros(shape=self.campaign.get_n_sub_campaigns(), dtype=np.bool)
        for i in range(self.campaign.get_n_sub_campaigns()):
            campaign_arm_mask = np.abs(self.last_sw_arm_rewards[i][arm_mask[i]][:, 0:self.sw_size // 2].sum(axis=1) -
                                       self.last_sw_arm_rewards[i][arm_mask[i]][:, -self.sw_size // 2:].sum(
                                           axis=1)) > self.cd_threshold
            campaign_mask[i] = campaign_arm_mask.any()

        if campaign_mask.any():
            campaign_mask = np.ones(shape=self.campaign.get_n_sub_campaigns(), dtype=bool)

        # print(campaign_mask.any())
        # print(self.t)
        return campaign_mask
