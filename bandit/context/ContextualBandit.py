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
     - full-context refers to a context (i.e. subset of space of features) that has its cardinality = n_features
     - a full-context is saved as a tuple of integers (e.g. (0, 0) for feature0 = 0 and feature1 = 0)
     - a context structure is defined as a list of list of full context. Basically, a context is composed by one or many
       full contexts
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
        self.bandit_dict: Dict[Tuple[int], DiscreteBandit] = {}  # data structure that maps full contexts into bandits
        self.frequency_context_generation: int = context_generation_frequency
        self.confidence: float = confidence
        self.n_features: int = n_features
        self.bandit_class = bandit_class
        self.bandit_kwargs = bandit_kwargs
        self.overall_bandit = bandit_class(**bandit_kwargs)
        self.context_structure: List[List[Tuple[int]]] = [[]]  #

        # Assign for each full context the general bandit that uses aggregated information
        for f in range(n_features):
            # Map integers into binary but saved inside a tuple of int
            full_context_str = format(f, "0" + str(n_features) + "b")
            full_context = tuple(map(int, full_context_str))

            self.bandit_dict[full_context] = self.overall_bandit

            # populate the context structure
            self.context_structure[0].append(full_context)

        # Set data structures to contain all information about features seen, pulled arms and collected rewards
        # (it is not information of the single bandit, but an aggregated information)
        self.features_to_index_dict: Dict[Tuple[int], List[int]] = defaultdict(list)
        self.pulled_arm_list: List[int] = []
        self.collected_rewards: List[float] = []
        self.day_t = 0
        self.t = 0

    def get_context_structure(self):
        return self.context_structure

    def pull_arm(self, full_context: Tuple[int]):
        """
        For a user with given full-context, choose the corresponding bandit with the full-context and pull from it

        :param full_context: a full-context which is a tuple with cardinality equal to n_features
        :return: the index of the arm pulled
        """
        for feature in full_context:
            assert feature in {0, 1}
        self.features_to_index_dict[full_context].append(self.t)
        return self.bandit_dict[full_context].pull_arm()

    def update(self, full_context: Tuple[int], pulled_arm: int, reward: float):
        """
        Update the contextual bandit by updating:
         - the time step
         - the ordered pulled arm list
         - the ordered collected rewards
         - the bandit corresponding to the given full-context

        :param full_context: the full-context of the previous pull_arm
        :param pulled_arm: the index of the pulled arm
        :param reward: the reward observed after pulling the arm
        """
        self.t += 1
        self.pulled_arm_list.append(pulled_arm)
        self.collected_rewards.append(reward)
        self.bandit_dict[full_context].update(pulled_arm, reward)

    def next_day(self):
        """
        Simulate the next day and if it is the day to re-generate the context, then generate the new context structure
        by using the greedy context generator. Then, re-create the bandits for each context in the new context structure
        and re-train them with the respective information (i.e. information only of users corresponding to that context)
        """
        self.day_t += 1

        if self.day_t % self.frequency_context_generation == 0:
            rewards_per_feature: Dict[Tuple[int], List[float]] = {}
            for feature, indices in self.features_to_index_dict.items():
                rewards_per_feature[feature] = list(np.array(self.collected_rewards)[indices])

            context_generator = ContextGenerator(self.n_features, self.confidence, rewards_per_feature)
            root_node = context_generator.generate_context_structure_tree({f for f in range(self.n_features)}, [])
            context_structure = [[]]
            self.context_structure = context_generator.get_context_structure_from_tree(root_node, [], context_structure)

            # Generate bandits
            for context in self.context_structure:
                self.bandit_dict = {}

                context_bandit = self.bandit_class(**self.bandit_kwargs)
                context_bandit = self._train_individual_bandit(context_bandit, context)

                for c in context:
                    self.bandit_dict[c] = context_bandit

    def _train_individual_bandit(self, bandit: DiscreteBandit, context: List[Tuple]) -> DiscreteBandit:
        """
        Train an individual bandit with the correct information (i.e. information only about the context given)

        :param bandit: the new empty bandit
        :param context: the context corresponding to that bandit
        :return: the trained bandit
        """
        pulled_arm_list = []
        collected_rewards = []
        for c in context:
            pulled_arm_list.extend(np.array(self.pulled_arm_list)[self.features_to_index_dict[c]])
            collected_rewards.extend(np.array(self.collected_rewards)[self.features_to_index_dict[c]])

        for i in range(len(pulled_arm_list)):
            bandit.update(pulled_arm_list[i], collected_rewards[i])
        return bandit
