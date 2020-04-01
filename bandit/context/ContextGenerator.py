from typing import Dict, Tuple, List, Set, Optional
import math
import numpy as np


class Node:
    """
    Class representing a general tree with integer information
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

    def __init__(self, n_features: int, confidence: float, min_context_to_rewards_dict: Dict[Tuple[int], List[float]]):
        """
        :param n_features: the number of features
        :param confidence: the confidence for the calculation of lower bound
        :param min_context_to_rewards_dict: a dictionary from min-context to list of rewards
        """
        self.n_features: int = n_features
        self.confidence: float = confidence
        self.min_context_to_rewards_dict: Dict[Tuple[int], List[float]] = min_context_to_rewards_dict

    @classmethod
    def get_hoeffding_lower_bound(cls, mean, confidence, cardinality):
        return mean - math.sqrt(-math.log(confidence) / (2 * cardinality))

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

    def generate_context_structure_tree(self, features_set: Set,
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
            rewards_0 = self.get_rewards_per_selected_features(selected_features + [(feature_idx, 0)])
            rewards_1 = self.get_rewards_per_selected_features(selected_features + [(feature_idx, 1)])
            lower_bound_p0 = self.get_hoeffding_lower_bound(len(rewards_0) / (len(rewards_0) + len(rewards_1)),
                                                            self.confidence, len(rewards_0) + len(rewards_1))
            lower_bound_p1 = self.get_hoeffding_lower_bound(len(rewards_1) / (len(rewards_0) + len(rewards_1)),
                                                            self.confidence, len(rewards_0) + len(rewards_1))
            lower_bound_v0 = self.get_hoeffding_lower_bound(sum(rewards_0) / len(rewards_0),
                                                            self.confidence, len(rewards_0))
            lower_bound_v1 = self.get_hoeffding_lower_bound(sum(rewards_1) / len(rewards_1),
                                                            self.confidence, len(rewards_1))
            context_value = lower_bound_p0 * lower_bound_v0 + lower_bound_p1 * lower_bound_v1
            context_values.append(context_value)
            lower_bound_current_context = self.get_hoeffding_lower_bound(
                (sum(rewards_0) + sum(rewards_1)) / (len(rewards_0) + len(rewards_1)),
                self.confidence, len(rewards_0) + len(rewards_1))
            lower_bound_current_context_list.append(lower_bound_current_context)

        for i in range(len(lower_bound_current_context_list)):
            assert lower_bound_current_context_list[0] == lower_bound_current_context_list[i]
        max_feature_idx = np.argmax(context_values)[0]

        if context_values[max_feature_idx] > lower_bound_current_context_list[0]:
            # If the best feature among all the "feature_set" is prominent, then select the best and call the recursive
            # function on the left child and right child
            features_set_child = features_set.copy()
            features_set_child.remove(max_feature_idx)
            left_node = self.generate_context_structure_tree(features_set_child,
                                                             selected_features + [(max_feature_idx, 0)])
            right_node = self.generate_context_structure_tree(features_set_child,
                                                              selected_features + [(max_feature_idx, 1)])

            node = Node(max_feature_idx)
            node.left = left_node
            node.right = right_node
            return node
        else:
            return None

    def get_rewards_per_selected_features(self, selected_features: List[Tuple[int, int]]) -> (List[float], List[float]):
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
        return rewards_per_selected_features

    def generate_context_per_features(self, selected_features: List[Tuple[int, int]]) -> List[Tuple[int]]:
        """
        Generate a context for the case (feature_idx=0, selected_features_left) as a tuple of integers

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features
                                  already selected
        :return: a context referring to (feature_idx=feature_value, selected_features)
        """
        context_per_features = []
        for f in range(self.n_features - 1 - len(selected_features)):
            combination_str = format(f, "0" + str(self.n_features - 1 - len(selected_features)) + "b")
            combination_values = list(map(int, combination_str))

            min_context = np.full(shape=self.n_features, fill_value=-1)
            for feature_idx, feature_value in selected_features:
                min_context[feature_idx] = feature_value
            min_context[min_context == -1] = combination_values

            context_per_features.append(tuple(min_context))
        return context_per_features
