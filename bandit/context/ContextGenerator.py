import itertools
from typing import Dict, Tuple, List, Set, Optional
import numpy as np

from bandit.discrete.DiscreteBandit import DiscreteBandit


class Node:
    """
    Class representing a node of a general tree with integer information
    """

    def __init__(self, data: int):
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.data: int = data


class ContextGenerator(object):
    """
    This class represents the greedy context generator algorithm which pseudo-code is:

    1 - start with feature_set = { all features }
    2 - for every feature, evaluate the inequalities of lower bounds of split feature versus the non-split case
    3 - select the feature that is more prominent (i.e. max evaluation) that is better than non-split case
    4 - repeat the algorithm from step 2 with the selected feature chosen equal to 0 and feature_set w/o feature chosen
    5 - repeat the algorithm from step 2 with the selected feature chosen equal to 1 and feature_set w/o feature chosen

    The idea is to create a tree that by visiting it, it gives you all the context in the context-structure chosen by
    the algorithm.

    There is the assumption that all features are binary
    """

    def __init__(self, n_features: int, confidence: float,
                 min_context_to_rewards_dict: Dict[Tuple[int, ...], List[float]],
                 min_context_to_pulled_arm_dict: Dict[Tuple[int, ...], List[int]],
                 bandit_class: DiscreteBandit.__class__, **bandit_kwargs):
        """
        :param n_features: the number of features
        :param confidence: the confidence for the calculation of lower bound
        :param min_context_to_rewards_dict: a dictionary from min-context to list of rewards
        """
        self.n_features: int = n_features
        self.confidence: float = confidence
        self.max_arm_value: float = np.max(bandit_kwargs["arm_values"])
        self.min_context_to_rewards_dict: Dict[Tuple[int, ...], List[float]] = min_context_to_rewards_dict
        self.min_context_to_pulled_arm_dict: Dict[Tuple[int, ...], List[int]] = min_context_to_pulled_arm_dict
        self.bandit_class: DiscreteBandit.__class__ = bandit_class
        self.bandit_kwargs = bandit_kwargs

    @classmethod
    def get_hoeffding_lower_bound(cls, mean, confidence, cardinality):
        if cardinality == 0:
            return -np.inf
        return mean - np.sqrt(-np.log(confidence) / (2 * cardinality))

    def get_context_structure_from_tree(self, node: Node, current_context: List[Tuple],
                                        context_structure: List[List[Tuple]]) -> List[List[Tuple]]:
        """
        Recursive function that populates the data structure "context_structure" passed by visiting the tree generated
        by "generate_context_structure_tree". At each leaf, it adds a new context into the "context_structure"

        :param node: the root of the tree or a sub-root of the sub-tree
        :param current_context: the actual context that is build by passing through node after node
        :param context_structure: the data structure that has to be populated by the function
        :return: the reference of the context_structure given at the start of the recursive function
        """
        if node is None:
            temp_context = self.generate_context_per_features(current_context)
            context_structure.append(temp_context)
            return context_structure

        if node.left is None and node.right is None:
            # Leaf reached: create the context obtained from the root to the leaf and add it to the context structure
            left_context = self.generate_context_per_features(current_context + [(node.data, 1)])
            right_context = self.generate_context_per_features(current_context + [(node.data, 0)])

            context_structure.append(left_context)
            context_structure.append(right_context)
            return context_structure

        self.get_context_structure_from_tree(node.left, current_context + [(node.data, 0)], context_structure)
        self.get_context_structure_from_tree(node.right, current_context + [(node.data, 1)], context_structure)
        return context_structure

    def continue_generation_context_structure_tree(self, node: Node, feature_set: List[int],
                                                   feature_selected: List[Tuple[int, int]]) -> Node:
        """

        :param node:
        :param feature_set:
        :param feature_selected:
        :return:
        """
        if node is None:
            new_node = self.generate_context_structure_tree(feature_set, feature_selected)
            return new_node
        feature_set_child = feature_set.copy()
        feature_set_child.remove(node.data)
        left_child = self.continue_generation_context_structure_tree(node.left, feature_set_child,
                                                                     feature_selected + [(node.data, 0)])
        right_child = self.continue_generation_context_structure_tree(node.left, feature_set_child,
                                                                      feature_selected + [(node.data, 1)])
        node.left = left_child
        node.right = right_child
        return node

    def generate_context_structure_tree(self, features_set: List,
                                        selected_features: List[Tuple[int, int]]) -> Optional[Node]:
        """
        Generates a tree containing information about the context structure chosen. By following the greedy algorithm,
        based on the comparison between split-case and non-split case, it chooses at each level, if the feature has to
        be split or not. It is a recursive function that returns the root of the tree

        :param features_set: the initial set of features which is commonly the set containing all features (indices)
        :param selected_features: the feature selected during the process (in case you go left, the feature selected are
                                  set to 0, while 1 if you go right on the tree)
        :return: the root of the tree
        """
        if len(features_set) == 0:
            return None

        context_values = []
        lower_bound_current_context_list = []

        for feature_idx in features_set:
            # Collect rewards and best rewards for each context (c_0, c_1 and c_total)
            rewards_0 = self.get_rewards_per_selected_features(selected_features + [(feature_idx, 0)])
            rewards_1 = self.get_rewards_per_selected_features(selected_features + [(feature_idx, 1)])
            best_rewards_0 = self.get_best_rewards_by_training_bandit(selected_features + [(feature_idx, 0)])
            best_rewards_1 = self.get_best_rewards_by_training_bandit(selected_features + [(feature_idx, 1)])
            best_rewards_total = self.get_best_rewards_by_training_bandit(selected_features)

            # Max-min normalize the best rewards (min = 0, max = max(all_rewards)
            best_rewards_0 = np.array(best_rewards_0) / self.max_arm_value
            best_rewards_1 = np.array(best_rewards_1) / self.max_arm_value
            best_rewards_total = np.array(best_rewards_total) / self.max_arm_value

            lower_bound_p0 = self.get_hoeffding_lower_bound(len(rewards_0) / (len(rewards_0) + len(rewards_1)),
                                                            self.confidence, len(rewards_0) + len(rewards_1))
            lower_bound_p1 = self.get_hoeffding_lower_bound(len(rewards_1) / (len(rewards_0) + len(rewards_1)),
                                                            self.confidence, len(rewards_0) + len(rewards_1))

            lower_bound_v0 = self.get_hoeffding_lower_bound(np.mean(best_rewards_0),
                                                            self.confidence, len(best_rewards_0))
            lower_bound_v1 = self.get_hoeffding_lower_bound(np.mean(best_rewards_1),
                                                            self.confidence, len(best_rewards_1))

            context_value = lower_bound_p0 * lower_bound_v0 + lower_bound_p1 * lower_bound_v1
            context_values.append(context_value)

            lower_bound_current_context = self.get_hoeffding_lower_bound(np.mean(best_rewards_total),
                                                                         self.confidence, len(best_rewards_total))
            lower_bound_current_context_list.append(lower_bound_current_context)

        for i in range(len(lower_bound_current_context_list)):
            assert lower_bound_current_context_list[0] == lower_bound_current_context_list[i]
        max_feature_idx = int(np.argmax(context_values))

        if context_values[max_feature_idx] > lower_bound_current_context_list[0]:
            # If the best feature among all the "feature_set" is prominent, then select the best and call the recursive
            # function on the left child and right child
            features_set_child = features_set.copy()
            features_set_child.remove(features_set[max_feature_idx])
            left_node = self.generate_context_structure_tree(features_set_child,
                                                             selected_features + [(features_set[max_feature_idx], 0)])
            right_node = self.generate_context_structure_tree(features_set_child,
                                                              selected_features + [(features_set[max_feature_idx], 1)])

            node = Node(features_set[max_feature_idx])
            node.left = left_node
            node.right = right_node
            return node
        else:
            return None

    def get_best_rewards_by_training_bandit(self, selected_features: List[Tuple[int, int]]) -> List[float]:
        """
        # TODO: comments on context classes
        :param selected_features:
        :return:
        """
        bandit: DiscreteBandit = self.bandit_class(**self.bandit_kwargs)
        rewards = self.get_rewards_per_selected_features(selected_features)
        pulled_arms = self.get_pulled_arms_per_selected_features(selected_features)
        for i in range(len(pulled_arms)):
            arm = pulled_arms[i]
            reward = rewards[i]
            bandit.update(arm, reward)
        best_rewards = bandit.rewards_per_arm[bandit.get_optimal_arm()]
        return best_rewards.copy()

    def get_rewards_per_selected_features(self, selected_features: List[Tuple[int, int]]) -> List[float]:
        """
        Returns the rewards given a list of selected features

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features already fixed
        :return: a tuple of reward of splitting with split-feature=0 and reward of splitting with split-feature=1
        """
        context = self.generate_context_per_features(selected_features)

        # Generate the rewards for the selected features
        rewards_per_selected_features = []
        for min_context in context:
            rewards_per_selected_features.extend(self.min_context_to_rewards_dict[min_context])
        return rewards_per_selected_features.copy()

    def get_pulled_arms_per_selected_features(self, selected_features: List[Tuple[int, int]]) -> List[int]:
        """

        :param selected_features:
        :return:
        """
        context = self.generate_context_per_features(selected_features)

        # Generate the rewards for the selected features
        pulled_arms_per_selected_features = []
        for min_context in context:
            pulled_arms_per_selected_features.extend(self.min_context_to_pulled_arm_dict[min_context])
        return pulled_arms_per_selected_features.copy()

    def generate_context_per_features(self, selected_features: List[Tuple[int, int]]) -> List[Tuple[int, ...]]:
        """
        Generate a context for the selected features as a tuple of integers

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features
                                  already selected
        :return: a context referring to (feature_idx=feature_value, selected_features)
        """
        context_per_features = []
        if self.n_features == len(selected_features):
            min_context = np.full(shape=self.n_features, fill_value=-1)
            for feature_idx, feature_value in selected_features:
                min_context[feature_idx] = feature_value
            if len(np.where(min_context == -1)[0]) > 0:
                print("no")
            context_per_features.append(tuple(min_context))
            return context_per_features

        for combination in list(itertools.product([0, 1], repeat=self.n_features-len(selected_features))):
            min_context = np.full(shape=self.n_features, fill_value=-1)
            for feature_idx, feature_value in selected_features:
                min_context[feature_idx] = feature_value
            min_context[min_context == -1] = combination

            context_per_features.append(tuple(min_context))
        return context_per_features
