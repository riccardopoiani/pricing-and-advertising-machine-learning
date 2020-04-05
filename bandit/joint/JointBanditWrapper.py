from abc import ABC
from typing import List

from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.IJointBandit import IJointBandit


class JointBanditWrapper(IJointBandit, ABC):
    """
    General class for the problem of jointly optimizing the pricing and the advertising strategy,
    based on wrapping a combinatorial bandit for advertising and multiple bandits for learning
    the price for each sub-campaign.
    """

    def __init__(self, ads_learner: CombinatorialBandit,
                 price_learner: List[DiscreteBandit],
                 campaign: Campaign):
        assert len(price_learner) == campaign.get_n_sub_campaigns()

        super().__init__(campaign=campaign)
        self.ads_learner: CombinatorialBandit = ads_learner
        self.price_learner: List[DiscreteBandit] = price_learner

    # Pull methods

    def pull_price(self, user_class: int) -> int:
        return self.price_learner[user_class].pull_arm()

    def pull_budget(self) -> List[int]:
        return self.ads_learner.pull_arm()

    # Update methods

    def update_price(self, user_class, pulled_arm, observed_reward) -> None:
        self.price_learner[user_class].update(pulled_arm, observed_reward)

    # Getter

    def get_reward_per_sub_campaign(self) -> List[List[float]]:
        return [learner.collected_rewards for learner in self.price_learner]
