from typing import List

from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.AdValueStrategy import AdValueStrategy
from bandit.joint.IJointBandit import IJointBandit


class JointBanditDiscriminatoryImproved(IJointBandit):
    """
    An improved version of discriminatory joint bandit for the problem for the problem of jointly optimizing
    the pricing and the advertising strategy, based on wrapping a combinatorial bandit for advertising
    and multiple bandits for learning the price for each sub-campaign.

    The calculation of the estimated value per clicks depends on the AdValueStrategy object and another core parameter
    is is_learn_visits that chooses if the advertising learner learns visits or visits*value per clicks.
    """

    def __init__(self, campaign: Campaign, ads_learner: CombinatorialBandit, price_learner: List[DiscreteBandit],
                 ad_value_strategy: AdValueStrategy):
        assert len(price_learner) == campaign.get_n_sub_campaigns()

        super().__init__(campaign=campaign)
        self.ads_learner: CombinatorialBandit = ads_learner
        self.price_learner: List[DiscreteBandit] = price_learner
        self.n_arms = price_learner[0].n_arms
        self.ad_value_strategy = ad_value_strategy

        self.estimated_ad_value = None

    # Pull methods

    def pull_price(self, user_class: int) -> int:
        return self.price_learner[user_class].pull_arm()

    def pull_budget(self) -> List[int]:
        return self.ads_learner.pull_arm(self.estimated_ad_value)

    # Update methods

    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        super(JointBanditDiscriminatoryImproved, self).update_price(user_class, pulled_arm, observed_reward)

        self.price_learner[user_class].update(pulled_arm, observed_reward)

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        super(JointBanditDiscriminatoryImproved, self).update_budget(pulled_arm_list, n_visits)

        # Compute the value of the ad
        opt_rewards_per_subcampaign = []
        for sub_idx in range(self.campaign.get_n_sub_campaigns()):
            optimal_arm = self.price_learner[sub_idx].get_optimal_arm()
            optimal_rewards = self.price_learner[sub_idx].rewards_per_arm[optimal_arm]
            opt_rewards_per_subcampaign.append(optimal_rewards)
        self.estimated_ad_value = self.ad_value_strategy.get_estimated_value_per_clicks(
            opt_rewards_per_subcampaign,
            day_t=self.day_t)

        # Update the model
        self.ads_learner.update(pulled_arm=pulled_arm_list, observed_reward=n_visits)
