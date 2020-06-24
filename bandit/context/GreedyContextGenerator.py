from typing import Tuple, List, Optional

import numpy as np

from bandit.context.AbstractContextGenerator import Node, AbstractContextGenerator


class GreedyContextGenerator(AbstractContextGenerator):
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

    def get_context_structure(self, old_cst_root: Node) -> (Optional[Node], List[List[Tuple]]):
        """
        Update the old context structure tree in order to allow only disaggregation of features using the greedy
        context generator algorithm

        :param old_cst_root: old context structure tree
        :return: new context structure in list format
        """
        feature_set = list(np.arange(0, self.n_features))
        new_cst_root = self._update_context_structure_tree_rec(old_cst_root, feature_set, [])

        context_structure = []
        return new_cst_root, self._get_context_structure_from_tree_rec(new_cst_root, [], context_structure)

    def _update_context_structure_tree_rec(self, node: Node, feature_set: List[int],
                                           feature_selected: List[Tuple[int, int]]) -> Node:
        """
        Recursive function that updates the old context structure tree (i.e. node variable) in order to avoid
        aggregation and allow only disaggregation of the context structure

        :param node: the current context structure tree that has to be updated
        :param feature_set: list of indices of features over doing the
        :param feature_selected: list of features already selected (
        :return:
        """
        if node is None:
            new_node = self._generate_context_structure_tree(feature_set, feature_selected)
            return new_node
        feature_set_child = feature_set.copy()
        feature_set_child.remove(node.data)
        left_child = self._update_context_structure_tree_rec(node.left, feature_set_child,
                                                             feature_selected + [(node.data, 0)])
        right_child = self._update_context_structure_tree_rec(node.left, feature_set_child,
                                                              feature_selected + [(node.data, 1)])
        node.left = left_child
        node.right = right_child
        return node

    def _generate_context_structure_tree(self, features_set: List,
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
            rewards_0 = self._get_rewards_per_selected_features(selected_features + [(feature_idx, 0)])
            rewards_1 = self._get_rewards_per_selected_features(selected_features + [(feature_idx, 1)])
            best_rewards_0 = self._get_best_rewards_by_training_bandit(selected_features +
                                                                       [(feature_idx, 0)])
            best_rewards_1 = self._get_best_rewards_by_training_bandit(selected_features +
                                                                       [(feature_idx, 1)])
            best_rewards_total = self._get_best_rewards_by_training_bandit(selected_features)

            lower_bound_p0 = self._get_hoeffding_lower_bound(len(rewards_0) / (len(rewards_0) + len(rewards_1)),
                                                             self.confidence, len(rewards_0) + len(rewards_1))
            lower_bound_p1 = self._get_hoeffding_lower_bound(len(rewards_1) / (len(rewards_0) + len(rewards_1)),
                                                             self.confidence, len(rewards_0) + len(rewards_1))

            lower_bound_v0 = self._get_hoeffding_lower_bound(np.mean(best_rewards_0),
                                                             self.confidence, len(best_rewards_0),
                                                             np.max(best_rewards_0))
            lower_bound_v1 = self._get_hoeffding_lower_bound(np.mean(best_rewards_1),
                                                             self.confidence, len(best_rewards_1),
                                                             np.max(best_rewards_1))

            context_value = lower_bound_p0 * lower_bound_v0 + lower_bound_p1 * lower_bound_v1
            context_values.append(context_value)

            lower_bound_current_context = self._get_hoeffding_lower_bound(np.mean(best_rewards_total),
                                                                          self.confidence, len(best_rewards_total),
                                                                          np.max(best_rewards_total))
            lower_bound_current_context_list.append(lower_bound_current_context)

        for i in range(len(lower_bound_current_context_list)):
            assert lower_bound_current_context_list[0] == lower_bound_current_context_list[i]
        max_feature_idx = int(np.argmax(context_values))

        if context_values[max_feature_idx] > lower_bound_current_context_list[0]:
            # If the best feature among all the "feature_set" is prominent, then select the best and call the recursive
            # function on the left child and right child
            features_set_child = features_set.copy()
            features_set_child.remove(features_set[max_feature_idx])
            left_node = self._generate_context_structure_tree(features_set_child,
                                                              selected_features + [(features_set[max_feature_idx], 0)])
            right_node = self._generate_context_structure_tree(features_set_child,
                                                               selected_features + [(features_set[max_feature_idx], 1)])

            node = Node(features_set[max_feature_idx])
            node.left = left_node
            node.right = right_node
            return node
        else:
            return None
