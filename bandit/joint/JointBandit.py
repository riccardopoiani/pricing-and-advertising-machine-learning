import numpy as np

from typing import List

from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.IJointBandit import IJointBandit


class JointBandit(IJointBandit):

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

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        expected_rewards = [np.array(learner.collected_rewards).mean() if len(learner.collected_rewards) > 0
                            else 0 for learner in self.price_learner]
        observed_rewards = np.array(n_visits) * np.array(expected_rewards)
        self.collected_total_rewards.append(observed_rewards.sum())
        self.ads_learner.update(pulled_arm=pulled_arm_list, observed_reward=observed_rewards)

    # Getter

    def get_reward_per_sub_campaign(self) -> List[List[float]]:
        return [learner.collected_rewards for learner in self.price_learner]

