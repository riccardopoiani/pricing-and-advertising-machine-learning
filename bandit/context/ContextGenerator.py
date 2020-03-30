from typing import Dict, Tuple, List, Set, Optional
import math
import numpy as np


class Node:
    def __init__(self, data: int):
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.data: int = data


class ContextGenerator(object):
    def __init__(self, n_features: int, confidence: float, rewards_per_context: Dict[Tuple[int], List[float]]):
        self.n_features: int = n_features
        self.confidence: float = confidence
        self.rewards_per_context: Dict[Tuple[int], List[float]] = rewards_per_context

    @classmethod
    def get_hoeffding_lower_bound(cls, mean, confidence, cardinality):
        return mean - math.sqrt(-math.log(confidence) / (2 * cardinality))

    def get_context_structure_from_tree(self, node: Node, current_context: List[Tuple],
                                        context_structure: List[List[Tuple]]) -> List[List[Tuple]]:
        if node is None:
            return context_structure

        if node.left is None and node.right is None:
            left_current_context = current_context + [(node.data, 0)]
            right_current_context = current_context + [(node.data, 1)]

            left_contexts = []
            right_contexts = []
            for f in range(self.n_features - len(left_current_context)):
                context_str = format(f, "0" + str(self.n_features - len(left_current_context)) + "b")
                context = list(map(int, context_str))
                context_left = [-1] * self.n_features
                context_right = [-1] * self.n_features

                for idx, value in left_current_context:
                    context_left[idx] = value

                for idx, value in right_current_context:
                    context_right[idx] = value

                for i in range(len(context_left)):
                    if context_left[i] == -1:
                        value = context.pop(0)
                        context_left[i] = value
                        context_right[i] = value
                context_left = tuple(context_left)
                context_right = tuple(context_right)
                left_contexts.append(context_left)
                right_contexts.append(context_right)
            context_structure.append(left_contexts)
            context_structure.append(right_contexts)
            return context_structure

        left_current_context = current_context + [(node.data, 0)]
        right_current_context = current_context + [(node.data, 1)]

        self.get_context_structure_from_tree(node.left, left_current_context, context_structure)
        self.get_context_structure_from_tree(node.right, right_current_context, context_structure)
        return context_structure

    def generate_context_structure_tree(self, features_set: Set, features_selected: List) -> Optional[Node]:
        if len(features_set) == 0:
            return None

        context_values = []
        lower_bound_current_context_list = []

        for feature_idx in features_set:
            rewards0, rewards1 = self.get_rewards_per_feature(feature_idx, features_selected)
            lower_bound_p0 = self.get_hoeffding_lower_bound(len(rewards0) / (len(rewards0) + len(rewards1)),
                                                            self.confidence, len(rewards0) + len(rewards1))
            lower_bound_p1 = self.get_hoeffding_lower_bound(len(rewards1) / (len(rewards0) + len(rewards1)),
                                                            self.confidence, len(rewards0) + len(rewards1))
            lower_bound_v0 = self.get_hoeffding_lower_bound(sum(rewards0) / len(rewards0),
                                                            self.confidence, len(rewards0))
            lower_bound_v1 = self.get_hoeffding_lower_bound(sum(rewards1) / len(rewards1),
                                                            self.confidence, len(rewards1))
            context_value = lower_bound_p0 * lower_bound_v0 + lower_bound_p1 * lower_bound_v1
            context_values.append(context_value)
            lower_bound_current_context = self.get_hoeffding_lower_bound(
                (sum(rewards0) + sum(rewards1)) / (len(rewards0) + len(rewards1)),
                self.confidence, len(rewards0) + len(rewards1))
            lower_bound_current_context_list.append(lower_bound_current_context)

        for i in range(len(lower_bound_current_context_list)):
            assert lower_bound_current_context_list[0] == lower_bound_current_context_list[i]
        max_feature_idx = np.argmax(context_values)[0]

        if context_values[max_feature_idx] > lower_bound_current_context_list[0]:
            features_set_child = features_set.copy()
            features_set_child.remove(max_feature_idx)
            features_selected_child0 = features_selected.copy()
            features_selected_child0.append((max_feature_idx, 0))
            features_selected_child1 = features_selected.copy()
            features_selected_child1.append((max_feature_idx, 1))

            left_node = self.generate_context_structure_tree(features_set_child, features_selected_child0)
            right_node = self.generate_context_structure_tree(features_set_child, features_selected_child1)

            node = Node(max_feature_idx)
            node.left = left_node
            node.right = right_node
            return node
        else:
            return None

    def get_rewards_per_feature(self, feature_idx, filtered_features: List[Tuple[int]]) -> (List[float], List[float]):
        feature0 = []
        feature1 = []
        for f in range(self.n_features - 1 - len(filtered_features)):
            context_str = format(f, "0" + str(self.n_features - 1 - len(filtered_features)) + "b")
            context = list(map(int, context_str))
            context0 = [-1] * self.n_features
            for idx, value in filtered_features:
                context0[idx] = value
            context1 = context0.copy()
            context0[feature_idx] = 0
            context1[feature_idx] = 1

            for i in range(len(context0)):
                if context0[i] == -1:
                    value = context.pop(0)
                    context0[i] = value
                    context1[i] = value

            feature0.append(tuple(context0))
            feature1.append(tuple(context1))
        rewards0 = []
        rewards1 = []
        for feature in feature0:
            rewards0.extend(self.rewards_per_context[feature])

        for feature in feature1:
            rewards1.extend(self.rewards_per_context[feature])

        return rewards0, rewards1
