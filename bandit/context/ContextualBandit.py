import itertools
from collections import defaultdict
from typing import Dict, Tuple, List

import numpy as np

from bandit.context.AbstractContextGenerator import AbstractContextGenerator
from bandit.discrete.DiscreteBandit import DiscreteBandit


class ContextualBandit(object):
    """
    A bandit that exploits the features of the users in order to discriminate them and maximize the overall reward.
    It uses an offline context generation algorithm (greedy one) that creates a context structure (a partition
    of the feature space) and for each context inside it, there is a different bandit used to select the best arms for
    the specific user

    Notation:
     - min-context refers to a smallest context (i.e. context is subset of feature space) that has all features fixed
     - a min-context is saved as a tuple of integers (e.g. (0, 0) for feature0 = 0 and feature1 = 0)
     - a general context is defined as a list of min-context, therefore, a context structure is defined as a list of
       general context
    """

    def __init__(self, n_features: int, confidence: float, context_generation_frequency: int,
                 context_generator_class: AbstractContextGenerator.__class__, bandit_class: DiscreteBandit.__class__,
                 **bandit_kwargs):
        """
        :param n_features: number of features known for each user (has to be the same for every user)
        :param confidence: confidence of the context generation algorithm used to calculate lower bounds
        :param context_generation_frequency: the frequency (in day) for the context generation algorithm
        :param context_generator_class: the class of the context generator to use
        :param bandit_class: the class of the bandit to use
        :param bandit_kwargs: the parameters of the bandit class
        """
        # data structure that maps min contexts into bandits
        self.min_context_to_bandit_dict: Dict[Tuple[int, ...], DiscreteBandit] = {}
        self.frequency_context_generation: int = context_generation_frequency
        self.context_generator_class = context_generator_class
        self.confidence: float = confidence
        self.n_features: int = n_features
        self.bandit_class = bandit_class
        self.bandit_kwargs = bandit_kwargs
        self.context_structure: List[List[Tuple[int, ...]]] = [[]]
        self.context_structure_tree = None

        overall_bandit = bandit_class(**bandit_kwargs)
        # Assign for each min-context the general bandit that uses aggregated information
        for min_context in list(itertools.product([0, 1], repeat=self.n_features)):
            self.min_context_to_bandit_dict[min_context] = overall_bandit
            self.context_structure[0].append(min_context)

        # Set data structures to contain all information about features seen, pulled arms and collected rewards
        # (it is not information of the single bandit, but an aggregated information)
        self.min_context_to_index_dict: Dict[Tuple[int], List[int]] = defaultdict(list)
        self.pulled_arm_list: List[int] = []
        self.collected_rewards: List[float] = []
        self.day_t = 0
        self.t = 0

    def get_context_structure(self):
        return self.context_structure

    def pull_arm(self, min_context: Tuple[int, ...]):
        """
        For a user with given min-context, choose the corresponding bandit with the min-context and pull from it

        :param min_context: a min-context which is a tuple with cardinality equal to n_features
        :return: the index of the arm pulled
        """
        for feature in min_context:
            assert feature in {0, 1}
        return self.min_context_to_bandit_dict[min_context].pull_arm()

    def update(self, min_context: Tuple[int, ...], pulled_arm: int, reward: float):
        """
        Update the contextual bandit by updating:
         - the time step
         - the ordered pulled arm list
         - the ordered collected rewards
         - the bandit corresponding to the given min-context

        :param min_context: the min-context related to the pulled_arm
        :param pulled_arm: the index of the pulled arm
        :param reward: the reward observed after pulling the arm
        """
        self.min_context_to_index_dict[min_context].append(self.t)
        self.t += 1
        self.pulled_arm_list.append(pulled_arm)
        self.collected_rewards.append(reward)
        self.min_context_to_bandit_dict[min_context].update(pulled_arm, reward)

    def next_day(self):
        """
        Simulate the next day and if it is the day to re-generate the context, then generate the new context structure
        by using the greedy context generator. Then, re-create the bandits for each context in the new context structure
        and re-train them with the respective information (i.e. information only of users corresponding to that context)
        """
        self.day_t += 1

        if self.day_t % self.frequency_context_generation == 0:
            rewards_per_feature: Dict[Tuple[int], List[float]] = {}
            pulled_arms_per_feature: Dict[Tuple[int], List[int]] = {}
            for min_context, indices in self.min_context_to_index_dict.items():
                rewards_per_feature[min_context] = list(np.array(self.collected_rewards.copy())[indices])
                pulled_arms_per_feature[min_context] = list(np.array(self.pulled_arm_list.copy())[indices])

            context_generator = self.context_generator_class(self.n_features, self.confidence, rewards_per_feature,
                                                             pulled_arms_per_feature, self.bandit_class,
                                                             **self.bandit_kwargs)
            self.context_structure_tree, self.context_structure = context_generator.get_context_structure(
                self.context_structure_tree)

            # Generate bandits
            self.min_context_to_bandit_dict = {}
            for context in self.context_structure:

                context_bandit = self.bandit_class(**self.bandit_kwargs)
                context_bandit = self._train_individual_bandit(context_bandit, context)

                for min_context in context:
                    self.min_context_to_bandit_dict[min_context] = context_bandit

    def _train_individual_bandit(self, bandit: DiscreteBandit, context: List[Tuple]) -> DiscreteBandit:
        """
        Train an individual bandit with the correct information (i.e. information only about the context given)

        :param bandit: the new empty bandit
        :param context: the context corresponding to that bandit
        :return: the trained bandit
        """

        indices = []
        for min_context in context:
            indices.extend(self.min_context_to_index_dict[min_context])
        indices = np.sort(indices)
        pulled_arm_list = np.array(self.pulled_arm_list.copy())[indices]
        collected_rewards = np.array(self.collected_rewards.copy())[indices]

        for i in range(len(pulled_arm_list)):
            bandit.update(pulled_arm_list[i], collected_rewards[i])
        return bandit
