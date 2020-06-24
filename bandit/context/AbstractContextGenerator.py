import itertools
from abc import abstractmethod
from typing import Dict, Tuple, List, Optional
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


class AbstractContextGenerator(object):
    """
    This class represents an abstract context generator built on top of tree in order to represents a context structure.
    It contains many useful functions in common with generic context generator.
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
        assert "arm_values" in bandit_kwargs.keys(), "The parameter arm_values is not in bandit_kwargs"

        self.n_features: int = n_features
        self.confidence: float = confidence
        self.arm_values = bandit_kwargs["arm_values"]
        self.min_context_to_rewards_dict: Dict[Tuple[int, ...], List[float]] = min_context_to_rewards_dict
        self.min_context_to_pulled_arm_dict: Dict[Tuple[int, ...], List[int]] = min_context_to_pulled_arm_dict
        self.bandit_class: DiscreteBandit.__class__ = bandit_class
        self.bandit_kwargs = bandit_kwargs

    @abstractmethod
    def get_context_structure(self, old_cst_root: Optional[Node]) -> (Optional[Node], List[List[Tuple]]):
        """
        Update the context structure tree and if requested by the algorithm, use also the old context structure tree
        for updating

        :param old_cst_root: old context structure tree if necessary
        :return: new context structure tree based on greedy algorithm on leaf nodes
        """
        pass

    def _get_context_structure_from_tree_rec(self, node: Node, current_context: List[Tuple],
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
            temp_context = self._generate_context_per_features(current_context)
            context_structure.append(temp_context)
            return context_structure

        if node.left is None and node.right is None:
            # Leaf reached: create the context obtained from the root to the leaf and add it to the context structure
            left_context = self._generate_context_per_features(current_context + [(node.data, 1)])
            right_context = self._generate_context_per_features(current_context + [(node.data, 0)])

            context_structure.append(left_context)
            context_structure.append(right_context)
            return context_structure

        self._get_context_structure_from_tree_rec(node.left, current_context + [(node.data, 0)], context_structure)
        self._get_context_structure_from_tree_rec(node.right, current_context + [(node.data, 1)], context_structure)
        return context_structure

    def _generate_context_per_features(self, selected_features: List[Tuple[int, int]]) -> List[Tuple[int, ...]]:
        """
        Generate the context related to the selected features and return it as a list of tuple of integers where each
        integer represent the value assigned to the feature in that position

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features
                                  already selected
        :return: a context referring to the selected features represented by a list of tuple of integers
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

        for combination in list(itertools.product([0, 1], repeat=self.n_features - len(selected_features))):
            min_context = np.full(shape=self.n_features, fill_value=-1)
            for feature_idx, feature_value in selected_features:
                min_context[feature_idx] = feature_value
            min_context[min_context == -1] = combination

            context_per_features.append(tuple(min_context))
        return context_per_features

    def _get_best_rewards_by_training_bandit(self, selected_features: List[Tuple[int, int]]) -> List[float]:
        """
        Returns the best rewards related to the context defined by the list of selected features. The best rewards is
        the rewards related to the optimal arm and it is obtained by re-training offline the bandit with only the
        related information of that context and ask the bandit the optimal arm

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features already fixed
        :return: the list of best rewards related to the selected features and the index of the optimal arm
        """
        bandit: DiscreteBandit = self.bandit_class(**self.bandit_kwargs)
        rewards = self._get_rewards_per_selected_features(selected_features)
        pulled_arms = self._get_pulled_arms_per_selected_features(selected_features)
        for i in range(len(pulled_arms)):
            arm = pulled_arms[i]
            reward = rewards[i]
            bandit.update(arm, reward)
        optimal_arm = bandit.get_optimal_arm()
        best_rewards = bandit.rewards_per_arm[optimal_arm]
        return best_rewards.copy()

    def _get_rewards_per_selected_features(self, selected_features: List[Tuple[int, int]]) -> List[float]:
        """
        Returns the rewards related to the context defined by the list of selected features

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features already fixed
        :return: the list of rewards related to the selected features
        """
        context = self._generate_context_per_features(selected_features)

        # Generate the rewards for the selected features
        rewards_per_selected_features = []
        for min_context in context:
            rewards_per_selected_features.extend(self.min_context_to_rewards_dict[min_context])
        return rewards_per_selected_features.copy()

    def _get_pulled_arms_per_selected_features(self, selected_features: List[Tuple[int, int]]) -> List[int]:
        """
        Return the pulled arms related to the context defined by the list of selected features

        :param selected_features: the list of tuple of (feature_idx, value) that represents the features already fixed
        :return: the list of pulled arms related to the selected features
        """
        context = self._generate_context_per_features(selected_features)

        # Generate the rewards for the selected features
        pulled_arms_per_selected_features = []
        for min_context in context:
            pulled_arms_per_selected_features.extend(self.min_context_to_pulled_arm_dict[min_context])
        return pulled_arms_per_selected_features.copy()

    @classmethod
    def _get_hoeffding_lower_bound(cls, mean, confidence, cardinality, bound_normalize_multiplier=1) -> float:
        """
        Return the value of the lower bound given by Hoeffding

        :param mean: the mean/average of the random variable
        :param confidence: the desired probability of error of the bound
        :param cardinality: the cardinality of observations of the random variable
        :return: the value of the lower bound given by Hoeffding
        """
        if cardinality == 0:
            return -np.inf
        return mean - bound_normalize_multiplier * np.sqrt(-np.log(confidence) / (2 * cardinality))
