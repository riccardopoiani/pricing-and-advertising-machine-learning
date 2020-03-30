from collections import defaultdict
from typing import Dict, Tuple, List
import numpy as np

from bandit.context.ContextGenerator import ContextGenerator
from bandit.discrete.DiscreteBandit import DiscreteBandit


class ContextualBandit(object):

    def __init__(self, n_features: int, confidence: float, context_generation_frequency: int,
                 bandit_class: DiscreteBandit.__class__, **bandit_kwargs):
        self.bandit_dict: Dict[Tuple[int], DiscreteBandit] = {}
        self.features_to_index_dict: Dict[Tuple[int], List[int]] = defaultdict(list)
        self.frequency_context_generation = context_generation_frequency
        self.confidence = confidence
        self.n_features = n_features
        self.bandit_class = bandit_class
        self.bandit_kwargs = bandit_kwargs

        self.overall_bandit = bandit_class(**bandit_kwargs)
        self.context_structure = [[]]

        for f in range(n_features):
            context_str = format(f, "0" + str(n_features) + "b")
            context = tuple(map(int, context_str))
            self.bandit_dict[context] = self.overall_bandit
            self.context_structure[0].append(context)

        self.pulled_arm_list: List[int] = []
        self.collected_rewards: List[float] = []
        self.day_t = 0
        self.t = 0

    def pull_arm(self, features: Tuple[int]):
        for feature in features:
            assert feature in {0, 1}
        self.features_to_index_dict[features].append(self.t)
        return self.bandit_dict[features].pull_arm()

    def update(self, pulled_arm: int, reward: float):
        self.t += 1
        self.pulled_arm_list.append(pulled_arm)
        self.collected_rewards.append(reward)

    def next_day(self):
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
        pulled_arm_list = []
        collected_rewards = []
        for c in context:
            pulled_arm_list.extend(np.array(self.pulled_arm_list)[self.features_to_index_dict[c]])
            collected_rewards.extend(np.array(self.collected_rewards)[self.features_to_index_dict[c]])

        for i in range(len(pulled_arm_list)):
            bandit.update(pulled_arm_list[i], collected_rewards[i])
        return bandit