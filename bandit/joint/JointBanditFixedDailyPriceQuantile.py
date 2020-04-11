from typing import List

import numpy as np
from scipy.stats import norm

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.joint.IJointBandit import IJointBandit


class JointBanditFixedDailyPriceQuantile(IJointBandit):
    """
    Optimize the joint problem of advertising and pricing in the following way:
    - A price is decided for a whole day and all classes of users: data is collected
    and managed according to the sub-campaign they come from
    - The optimization procedure maximizes the profit by jointly maximizing the number of visit
    and ads value by considering each price at a time, and solving the optimization problem.
    In particular, samples from the number of visits and samples statistics
    for the advertising click are considered in order to trade-off exploration and exploitation:
    the best among these results is considered for the next day as best price and as the best
    budget to be invested
    """

    def __init__(self, campaign: Campaign, number_of_visit_model_list: List[DiscreteRegressor],
                 n_arms_profit: int, arm_profit: np.array, min_std: float):
        assert len(number_of_visit_model_list) == campaign.get_n_sub_campaigns()

        super().__init__(campaign=campaign)

        # Number of visit data structure
        self.number_of_visit_model_list: List[DiscreteRegressor] = number_of_visit_model_list
        self.collected_rewards_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_arm_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]

        # Pricing data structure
        self.n_arms_price: int = n_arms_profit
        self.arm_profit = arm_profit
        self.profit_arm_reward_list = [[[] for _ in range(n_arms_profit)] for _ in
                                       range(campaign.get_n_sub_campaigns())]
        self.min_std = min_std

        # Current data structure
        self.current_pricing_arm_idx = np.random.randint(low=0, high=n_arms_profit, size=1)
        self.curr_best_budget_idx: List[int] = [0 for _ in range(campaign.get_n_sub_campaigns())]
        self.current_budget_allocation = [0 for _ in range(campaign.get_n_sub_campaigns())]

        # Initializing randomly budget values
        for sub_index, model in enumerate(self.number_of_visit_model_list):
            sub_campaign_values = self.number_of_visit_model_list[sub_index].sample_distribution()
            self.campaign.set_sub_campaign_values(sub_index, sub_campaign_values)
        _, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
        self.curr_best_budget_idx = [np.where(self.campaign.get_budgets() == budget)[0][0] for budget in
                                     best_budgets]

    def pull_price(self, user_class: int) -> int:
        return self.current_pricing_arm_idx

    def pull_budget(self) -> List[int]:
        return self.curr_best_budget_idx

    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        super(JointBanditFixedDailyPriceQuantile, self).update_price(user_class, pulled_arm, observed_reward)

        self.profit_arm_reward_list[user_class][pulled_arm].append(observed_reward)

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        super(JointBanditFixedDailyPriceQuantile, self).update_budget(pulled_arm_list, n_visits)

        # Update data structure
        for i in range(self.campaign.get_n_sub_campaigns()):
            self.pulled_arm_sub_campaign[i].append(pulled_arm_list[i])
            self.collected_rewards_sub_campaign[i].append(n_visits[i])

        # Update model with new information
        for sub_index, model in enumerate(self.number_of_visit_model_list):
            model.fit_model(collected_rewards=self.collected_rewards_sub_campaign[sub_index],
                            pulled_arm_history=self.pulled_arm_sub_campaign[sub_index])

        # For all the sub-campaigns and profit-arms compute mean and std, and the quantile
        mean_ad_value = np.zeros(shape=(self.campaign.get_n_sub_campaigns(), self.n_arms_price))
        std_ad_value = np.zeros(shape=(self.campaign.get_n_sub_campaigns(), self.n_arms_price))
        percentile = 1 - (1 / (self.day_t + 1))

        for c in range(self.campaign.get_n_sub_campaigns()):
            for arm in range(self.n_arms_price):
                values = np.array(self.profit_arm_reward_list[c][arm])
                mean_ad_value[c][arm] = values.mean() if len(values) > 0 else self.arm_profit[arm]
                std_ad_value[c][arm] = values.std() if len(values) > 0 else self.min_std

        std_ad_value = np.where(std_ad_value < self.min_std, self.min_std, std_ad_value)

        estimated_ad_value = norm.ppf(q=percentile, loc=mean_ad_value, scale=std_ad_value)

        # Sample the number of visits for each sub-campaign
        sample_visit = np.zeros(
            shape=(self.campaign.get_n_sub_campaigns(), len(self.number_of_visit_model_list[0].arms)))
        for c in range(self.campaign.get_n_sub_campaigns()):
            sample_visit[c] = self.number_of_visit_model_list[c].sample_distribution()

        # Joint optimization of advertising and pricing
        best_arm_profit_idx = -1
        curr_max_profit = -1
        curr_best_budget_idx = [-1 for _ in range(self.campaign.get_n_sub_campaigns())]
        for profit_arm_index in range(self.n_arms_price):
            # Set campaign
            for sub_campaign_idx in range(self.campaign.get_n_sub_campaigns()):
                sub_campaign_visits = sample_visit[sub_campaign_idx]
                sub_campaign_values = sub_campaign_visits * estimated_ad_value[sub_campaign_idx][profit_arm_index]
                self.campaign.set_sub_campaign_values(sub_campaign_idx, sub_campaign_values)

            # Campaign optimization
            max_profit, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)
            if max_profit > curr_max_profit:
                curr_max_profit = max_profit
                curr_best_budget_idx = [np.where(self.campaign.get_budgets() == budget)[0][0] for budget in
                                        best_budgets]
                best_arm_profit_idx = profit_arm_index

        self.current_pricing_arm_idx = best_arm_profit_idx

        self.curr_best_budget_idx = curr_best_budget_idx
