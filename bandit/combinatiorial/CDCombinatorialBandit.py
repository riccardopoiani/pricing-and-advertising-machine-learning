import numpy as np

from typing import List

from advertising.data_structure import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit


class CDCombinatorialBandit(CombinatorialBandit):
    """
    Change Detection combinatorial bandit:
    """
    # TODO
    #  -check sw_size is an even number
    #  -convert pulled_superarm_after_detection from List to array
    #  -last_detection_time is a 3-D array! fix the pull_arm function
    #  -change fit_model parameters: the hystory lenght of pulled_arms/rewards vary depending on the campaign
    #  -add the bandit to the list in run_advertising_budget_optimizatio.py
    #  -add / change comments
    def __init__(self, campaign: Campaign, model_list: List[DiscreteRegressor], n_arms: int, gamma: float,
                 cd_threshold: float, sw_size: int):
        super().__init__(campaign=campaign, model_list=model_list)
        self.n_arms = n_arms
        self.gamma = gamma
        self.cd_threshold = cd_threshold
        self.sw_size = sw_size
        self.last_detection_time = np.zeros(3)
        self.pulled_superarm_after_detection = [[0 for _ in range(n_arms)]
                                                for _ in range(campaign.get_n_sub_campaigns())]
        self.last_sw_arm_rewards = np.zeros((campaign.get_n_sub_campaigns(), n_arms, sw_size))

        self.collected_rewards_sub_campaign_after_detection = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_arm_sub_campaign_after_detection = [[] for _ in range(campaign.get_n_sub_campaigns())]

    def pull_arm(self) -> List[int]:
        """
        Find the best allocation of budgets by optimizing the combinatorial problem of the campaign and then return
        the indices of the best budgets.
        The combinatorial problem is optimized given estimates provided only by the last data (amount specified
        by the sliding window)

        :return: the indices of the best budgets given the actual campaign
        """
        #arm_idx = np.zeros(3)
        #for i in range(self.campaign.get_n_sub_campaigns()):
        #    arm_idx[i] = (self.t - self.last_detection_time[i]) % np.math.floor(self.n_arms / self.gamma)

        #if arm_idx <= self.n_arms:
        #    return arm_idx

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

        #
        for i in range(self.campaign.get_n_sub_campaigns()):
            self.last_sw_arm_rewards[i][pulled_arm[i]][self.pulled_superarm_after_detection[i][pulled_arm[i]]] = reward[i]

            self.pulled_superarm_after_detection[i][pulled_arm[i]] += 1
            print(self.pulled_superarm_after_detection[i][pulled_arm[i]])
            self.collected_rewards_sub_campaign_after_detection[i].append(reward[i])
            self.pulled_arm_sub_campaign_after_detection[i].append(pulled_arm[i])

    def update(self, pulled_arm: List[int], reward: List[float]) -> None:
        """
        Update observations and models of the sub-campaign

        :param pulled_arm: list of indices of the pulled arms (i.e. superarm pulled)
        :param reward: list of observed reward for each pulled arm
        :return: None
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)

        for i in range(self.campaign.get_n_sub_campaigns()):
            if (self.pulled_superarm_after_detection[i][pulled_arm[i]]) >= self.sw_size:

                if self._change_detection(self.last_sw_arm_rewards[i][pulled_arm[i]]):
                    self.last_detection_time[i] = self.t - 1
                    for arm in range(self.n_arms):
                        print("reset")
                        print(t)
                        self.pulled_superarm_after_detection[i][arm] = 0

                    # remove all the appended elements to the subcampaign for which we encounter a detection
                    for _ in range(len(self.pulled_arm_sub_campaign_after_detection[i])):
                        del self.pulled_arm_sub_campaign_after_detection[i][0]
                        del self.collected_rewards_sub_campaign_after_detection[i][0]

        for sub_index, model in enumerate(self.model_list):
            model.fit_model(collected_rewards=self.collected_rewards_sub_campaign_after_detection[sub_index],
                            pulled_arm_history=self.pulled_arm_sub_campaign_after_detection[sub_index])

            # Update estimations of the values of the sub-campaigns
            sub_campaign_values = self.model_list[sub_index].sample_distribution()
            self.campaign.set_sub_campaign(sub_index, sub_campaign_values)

    def _change_detection(self, observations: np) -> int:
        first_half = 0
        second_half = 0

        for i in range(self.sw_size):
            if i < (self.sw_size/2):
                first_half += observations[i]
            else:
                second_half += observations[i]

        if (second_half - first_half) > self.cd_threshold:
            return True
        return False
