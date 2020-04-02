from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np

from bandit.context.ContextGenerator import ContextGenerator
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
                 bandit_class: DiscreteBandit.__class__, **bandit_kwargs):
        """
        :param n_features: number of features known for each user (has to be the same for every user)
        :param confidence: confidence of the context generation algorithm used to calculate lower bounds
        :param context_generation_frequency: the frequency (in day) for the context generation algorithm
        :param bandit_class: the class of the bandit to use
        :param bandit_kwargs: the parameters of the bandit class
        """
        # data structure that maps min contexts into bandits
        self.min_context_to_bandit_dict: Dict[Tuple[int], DiscreteBandit] = {}
        self.frequency_context_generation: int = context_generation_frequency
        self.confidence: float = confidence
        self.n_features: int = n_features
        self.bandit_class = bandit_class
        self.bandit_kwargs = bandit_kwargs
        self.overall_bandit = bandit_class(**bandit_kwargs)
        self.context_structure: List[List[Tuple[int]]] = [[]]  #

        # Assign for each min-context the general bandit that uses aggregated information
        for f in range(2**n_features):
            # Map integers into binary but saved inside a tuple of int
            full_context_str = format(f, "0" + str(n_features) + "b")
            full_context = tuple(map(int, full_context_str))

            self.min_context_to_bandit_dict[full_context] = self.overall_bandit

            # populate the context structure
            self.context_structure[0].append(full_context)

        # Set data structures to contain all information about features seen, pulled arms and collected rewards
        # (it is not information of the single bandit, but an aggregated information)
        self.min_context_to_index_dict: Dict[Tuple[int], List[int]] = defaultdict(list)
        self.pulled_arm_list: List[int] = []
        self.collected_rewards: List[float] = []
        self.day_t = 0
        self.t = 0

    def get_context_structure(self):
        return self.context_structure

    def pull_arm(self, min_context: Tuple[int]):
        """
        For a user with given min-context, choose the corresponding bandit with the min-context and pull from it

        :param min_context: a min-context which is a tuple with cardinality equal to n_features
        :return: the index of the arm pulled
        """
        for feature in min_context:
            assert feature in {0, 1}
        self.min_context_to_index_dict[min_context].append(self.t)
        return self.min_context_to_bandit_dict[min_context].pull_arm()

    def update(self, min_context: Tuple[int], pulled_arm: int, reward: float):
        """
        Update the contextual bandit by updating:
         - the time step
         - the ordered pulled arm list
         - the ordered collected rewards
         - the bandit corresponding to the given min-context

        :param min_context: the min-context of the previous pull_arm
        :param pulled_arm: the index of the pulled arm
        :param reward: the reward observed after pulling the arm
        """
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
            for feature, indices in self.min_context_to_index_dict.items():
                rewards_per_feature[feature] = list(np.array(self.collected_rewards)[indices])
                pulled_arms_per_feature[feature] = list(np.array(self.pulled_arm_list)[indices])

            context_generator = ContextGenerator(self.n_features, self.confidence, rewards_per_feature,
                                                 pulled_arms_per_feature, self.bandit_class, **self.bandit_kwargs)
            root_node = context_generator.generate_context_structure_tree([f for f in range(self.n_features)], [])
            context_structure = []
            self.context_structure = context_generator.get_context_structure_from_tree(root_node, [], context_structure)

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
        pulled_arm_list = []
        collected_rewards = []
        for min_context in context:
            pulled_arm_list.extend(np.array(self.pulled_arm_list)[self.min_context_to_index_dict[min_context]])
            collected_rewards.extend(np.array(self.collected_rewards)[self.min_context_to_index_dict[min_context]])

        for i in range(len(pulled_arm_list)):
            bandit.update(pulled_arm_list[i], collected_rewards[i])
        return bandit
