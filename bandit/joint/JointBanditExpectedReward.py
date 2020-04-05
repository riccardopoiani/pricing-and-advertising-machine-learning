import numpy as np

from typing import List

from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from bandit.joint.JointBanditWrapper import JointBanditWrapper


class JointBanditExpectedReward(JointBanditWrapper):
    """
    This method for the joint optimization feed the campaign with rewards composed of the
    number of visit multiplied by the empirical expected reward.
    """

    def __init__(self, ads_learner: CombinatorialBandit,
                 price_learner: List[DiscreteBandit],
                 campaign: Campaign):
        super().__init__(ads_learner=ads_learner,
                         price_learner=price_learner,
                         campaign=campaign)

    def update_budget(self, pulled_arm_list: List[int], n_visits: List[float]):
        expected_rewards = [np.array(learner.collected_rewards).mean() if len(learner.collected_rewards) > 0
                            else 0 for learner in self.price_learner]
        observed_rewards = np.array(n_visits) * np.array(expected_rewards)
        self.ads_learner.update(pulled_arm=pulled_arm_list, observed_reward=observed_rewards)

        # Update the current reward obtained from the model
        total_r = np.array([np.array(learner.collected_rewards).sum() for learner in self.price_learner]).sum()
        self.collected_total_rewards.append(total_r - np.array(self.collected_total_rewards).sum())