import unittest
import numpy as np

from bandit.context.ContextGenerator import GreedyContextGenerator
from bandit.discrete.UCB1Bandit import UCB1Bandit


class ContextGeneratorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        min_context_to_rewards_dict = {(0, 0): [100, 100, 200, 200, 200],
                                       (0, 1): [100, 100, 200, 200, 200],
                                       (1, 0): [100, 100, 200, 200, 200],
                                       (1, 1): [100, 100, 200, 200, 200]}
        min_context_to_pulled_arms_dict = {(0, 0): [3, 3, 2, 2, 2],
                                           (0, 1): [3, 3, 2, 2, 2],
                                           (1, 0): [2, 2, 3, 3, 3],
                                           (1, 1): [2, 2, 3, 3, 3]}

        self.context_generator_2 = GreedyContextGenerator(2, 0.95, min_context_to_rewards_dict,
                                                          min_context_to_pulled_arms_dict, UCB1Bandit, n_arms=10,
                                                          arm_values=np.linspace(0, 1, 10))

        min_context_to_rewards_dict = {(0, 0): [100, 100, 200, 200, 200],
                                       (0, 1): [100, 100, 200, 200, 200],
                                       (1, 0): [100, 100, 200, 200, 200, 300, 300, 300],
                                       (1, 1): [100, 100, 200, 200, 200, 50, 50, 50]}
        min_context_to_pulled_arms_dict = {(0, 0): [3, 3, 2, 2, 2],
                                           (0, 1): [3, 3, 2, 2, 2],
                                           (1, 0): [2, 2, 3, 3, 3, 1, 1, 1],
                                           (1, 1): [2, 2, 3, 3, 3, 1, 1, 1]}

        self.context_generator_2_1 = GreedyContextGenerator(2, 0.95, min_context_to_rewards_dict,
                                                            min_context_to_pulled_arms_dict, UCB1Bandit, n_arms=10,
                                                            arm_values=np.linspace(0, 1, 10))

    def test_generate_context_per_all_features(self):
        selected_features = [(0, 1,), (1, 0,)]
        context = self.context_generator_2._generate_context_per_features(selected_features)
        expected_context = [(1, 0)]

        self.assertTrue(expected_context == context)

    def test_generate_context_per_features(self):
        selected_features = [(0, 1,)]
        context = self.context_generator_2._generate_context_per_features(selected_features)
        expected_context = [(1, 0,), (1, 1,)]

        self.assertTrue(expected_context == context)

    def test_generate_context_per_no_features(self):
        selected_features = []
        context = self.context_generator_2._generate_context_per_features(selected_features)
        expected_context = [(0, 0,), (0, 1,), (1, 0,), (1, 1,)]

        self.assertTrue(expected_context == context)

    def test_get_rewards_per_selected_features(self):
        selected_features = [(0, 1,), (1, 0,)]
        rewards = self.context_generator_2._get_rewards_per_selected_features(selected_features)
        expected_rewards = self.context_generator_2.min_context_to_rewards_dict[(1, 0,)]
        self.assertTrue(rewards == expected_rewards)

        selected_features = []
        rewards = self.context_generator_2._get_rewards_per_selected_features(selected_features)
        expected_rewards = []
        for x in self.context_generator_2.min_context_to_rewards_dict.values():
            expected_rewards.extend(x)
        self.assertTrue(rewards == expected_rewards)

        selected_features = [(0, 1,)]
        rewards = self.context_generator_2._get_rewards_per_selected_features(selected_features)
        expected_rewards = self.context_generator_2.min_context_to_rewards_dict[(1, 0)] + \
                           self.context_generator_2.min_context_to_rewards_dict[(1, 1)]
        self.assertTrue(rewards == expected_rewards)

    def test_generate_context_structure_tree(self):
        feature_set = [0, 1]
        selected_features = []
        root = self.context_generator_2._generate_context_structure_tree(feature_set, selected_features)

        self.assertTrue(root.data == 0)
        self.assertTrue(root.left is None)
        self.assertTrue(root.right is None)

    def test_generate_context_structure_tree_2(self):
        feature_set = [0, 1]
        selected_features = []
        root = self.context_generator_2_1._generate_context_structure_tree(feature_set, selected_features)

        self.assertTrue(root.data == 1)
        self.assertTrue(root.left is None)
        self.assertTrue(root.right is not None)
        self.assertTrue(root.right.data == 0)

    def test_get_context_structure_tree(self):
        feature_set = [0, 1]
        selected_features = []
        root = self.context_generator_2._generate_context_structure_tree(feature_set, selected_features)
        context_structure = []
        context_structure = self.context_generator_2._get_context_structure_from_tree_rec(root, [], context_structure)
        self.assertTrue(context_structure == [[(1, 0), (1, 1)], [(0, 0), (0, 1)]])

    def test_get_context_structure_tree_2(self):
        feature_set = [0, 1]
        selected_features = []
        root = self.context_generator_2_1._generate_context_structure_tree(feature_set, selected_features)
        context_structure = []
        context_structure = self.context_generator_2_1._get_context_structure_from_tree_rec(root, [], context_structure)
        self.assertTrue(context_structure == [[(0, 0), (1, 0)], [(1, 1)], [(0, 1)]])
