from collections import defaultdict
from typing import List, Dict
from sympy.solvers.inequalities import solve_rational_inequalities
from sympy import Poly, FiniteSet
from sympy.abc import x

import numpy as np

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.AdValueStrategy import AdValueStrategy
from bandit.joint.IJointBandit import IJointBandit


class JointBanditBalanced(IJointBandit):
    """
    A balanced joint bandit for the problem for the problem of jointly optimizing the pricing and the advertising
    strategy, based on multiples combinatorial bandits for advertising and unique bandit for learning
    the price for each sub-campaign.
     - There is no discrimination of users
     - The distribution of users in the unique bandit is balanced based on the number of clicks estimated
     - The calculation of the estimated value per clicks depends on the AdValueStrategy object
    """

    def __init__(self, campaign: Campaign, arm_values: np.ndarray, price_learner_class: DiscreteBandit.__class__,
                 price_learner_kwargs, number_of_visit_model_list: List[DiscreteRegressor],
                 ad_value_strategy: AdValueStrategy):
        super().__init__(campaign)
        assert len(number_of_visit_model_list) == campaign.get_n_sub_campaigns()

        # General problem data
        self.arm_values: np.ndarray = arm_values

        # Data structure to save overall data
        self.collected_rewards_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]
        self.pulled_arm_sub_campaign: List[List] = [[] for _ in range(campaign.get_n_sub_campaigns())]

        self.value_per_clicks_per_price_idx: Dict[int, List[float]] = {}
        for arm_idx in range(len(self.arm_values)):
            rewards_per_subcampaign = [[] for _ in range(self.campaign.get_n_sub_campaigns())]

            self.value_per_clicks_per_price_idx[arm_idx] = ad_value_strategy. \
                get_estimated_value_per_clicks(rewards_per_subcampaign, self.day_t)

        self.rewards_per_arm_per_user_class: Dict[int, List[List[float]]] = defaultdict(list)
        for i in range(self.campaign.get_n_sub_campaigns()):
            for j in range(len(self.arm_values)):
                self.rewards_per_arm_per_user_class[i].append([])

        # Learners
        self.price_bandit_class = price_learner_class
        self.price_bandit_kwargs = price_learner_kwargs

        self.unique_price_learner: DiscreteBandit = self.price_bandit_class(**self.price_bandit_kwargs)
        self.number_of_visit_model_list: List[DiscreteRegressor] = number_of_visit_model_list
        self.ad_value_strategy: AdValueStrategy = ad_value_strategy

    def pull_price(self, user_class: int) -> int:
        return self.unique_price_learner.pull_arm()

    def pull_budget(self) -> List[int]:
        max_best_clicks = 0
        max_best_budgets = None

        for arm_idx in range(len(self.arm_values)):
            subcampaign_values = self.campaign.get_sub_campaigns()
            arm_value_campaign = Campaign(self.campaign.get_n_sub_campaigns(), self.campaign.get_cum_budget(),
                                          len(self.campaign.get_budgets()))
            for sub_idx in range(len(subcampaign_values)):
                arm_value_campaign.set_sub_campaign_values(sub_idx, subcampaign_values[sub_idx] *
                                                           self.value_per_clicks_per_price_idx[arm_idx][sub_idx])
            max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(self.campaign)

            if max_best_clicks < max_clicks:
                max_best_clicks = max_clicks
                max_best_budgets = best_budgets

        # Balance price learner with number of clicks distribution estimation
        budget_value_to_index = {value: i for i, value in enumerate(self.campaign.get_budgets())}
        estimated_clicks = np.array([self.campaign.get_sub_campaigns()[sub_idx, budget_value_to_index[budget]]
                                     for sub_idx, budget in enumerate(max_best_budgets)])
        user_probabilities = estimated_clicks / np.sum(estimated_clicks)

        self.unique_price_learner: DiscreteBandit = self.price_bandit_class(**self.price_bandit_kwargs)

        for arm_idx in range(len(self.arm_values)):
            rewards_per_subcampaign = []
            for sub_idx in range(self.campaign.get_n_sub_campaigns()):
                rewards_per_subcampaign.append(self.rewards_per_arm_per_user_class[sub_idx][arm_idx])
            rewards_len = [len(rewards) for rewards in rewards_per_subcampaign]

            inequalities = []
            for i in range(len(user_probabilities)):
                inequalities.append(((Poly(user_probabilities[i] * x - rewards_len[i]), Poly(1, x)), '<='))
            solution = solve_rational_inequalities([inequalities])

            if type(solution) is FiniteSet:
                solution = np.max(int(solution[0]), 0)
            else:
                solution = np.max(int(solution.end), 0)

            balanced_rewards_len = np.array(np.floor(solution * user_probabilities), dtype=int)

            for sub_idx in range(self.campaign.get_n_sub_campaigns()):
                if balanced_rewards_len[sub_idx] <= 0:
                    continue

                sampled_rewards = np.random.choice(rewards_per_subcampaign[sub_idx],
                                                   size=balanced_rewards_len[sub_idx],
                                                   replace=False)
                for reward in sampled_rewards:
                    self.unique_price_learner.update(arm_idx, reward)

        return [budget_value_to_index[budget] for budget in max_best_budgets]

    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        super(JointBanditBalanced, self).update_price(user_class, pulled_arm, observed_reward)

        self.rewards_per_arm_per_user_class[user_class][pulled_arm].append(observed_reward)
        self.unique_price_learner.update(pulled_arm, observed_reward)

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        super(JointBanditBalanced, self).update_budget(pulled_arm_list, n_visits)

        # Update data structure
        for i in range(self.campaign.get_n_sub_campaigns()):
            self.pulled_arm_sub_campaign[i].append(pulled_arm_list[i])
            self.collected_rewards_sub_campaign[i].append(n_visits[i])

        # Update model with new information
        for sub_index, model in enumerate(self.number_of_visit_model_list):
            model.fit_model(collected_rewards=self.collected_rewards_sub_campaign[sub_index],
                            pulled_arm_history=self.pulled_arm_sub_campaign[sub_index])
            # Update estimations of the values of the sub-campaigns
            self.campaign.set_sub_campaign_values(sub_index,
                                                  self.number_of_visit_model_list[sub_index].sample_distribution())

        # Set value per clicks for each arm
        for arm_idx in range(len(self.arm_values)):
            rewards_per_subcampaign = [[] for _ in range(self.campaign.get_n_sub_campaigns())]
            for user_class, rewards_per_arm in self.rewards_per_arm_per_user_class.items():
                rewards_per_subcampaign[user_class].extend(rewards_per_arm[arm_idx])

            self.value_per_clicks_per_price_idx[arm_idx] = self.ad_value_strategy. \
                get_estimated_value_per_clicks(rewards_per_subcampaign, self.day_t)
