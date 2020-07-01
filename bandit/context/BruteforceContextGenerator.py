import itertools
from typing import Optional, List, Tuple

import numpy as np

from bandit.context.AbstractContextGenerator import AbstractContextGenerator, Node
from bandit.discrete.DiscreteBandit import DiscreteBandit


def generate_partition(collection):
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in generate_partition(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
        # put `first` in its own subset
        yield [[first]] + smaller


class BruteforceContextGenerator(AbstractContextGenerator):
    """
    A simple context generator that compares all possible partitions and select the one with the highest value
    """

    def get_context_structure(self, old_cst_root: Node) -> (Optional[Node], List[List[Tuple]]):
        """
        Update the context structure tree by looking at every possible combinations of feature without looking at the
        old context structure

        :param old_cst_root: old context structure tree
        :return: new context structure tree based on greedy algorithm on leaf nodes
        """
        # list all possible partitions
        min_context_list = []
        for i in itertools.product([0, 1], repeat=self.n_features):
            min_context_list.append(i)

        partitions = []
        for partition in generate_partition(min_context_list):
            partitions.append(partition)

        # For each partition, compute the expected value
        partition_values = []
        max_best_rewards_p_list = []
        for partition in partitions:
            partition_value = 0
            len_rewards_list = []
            mean_best_rewards_list = []
            max_best_rewards_list = []
            len_best_rewards_list = []

            # Compute all necessary values for each context inside the partition
            for context in partition:
                context_best_rewards = self._get_best_rewards_by_context_and_training_bandit(context)

                len_rewards_list.append(len(self._get_rewards_per_context(context)))
                mean_best_rewards_list.append(np.mean(context_best_rewards))
                max_best_rewards_list.append(np.max(context_best_rewards))
                len_best_rewards_list.append(len(context_best_rewards))

            # Compute the partition value by summing the lower bound probabilities times the lower bound value of each
            # context using Hoeffding bound
            sum_rewards_length = np.sum(len_rewards_list)
            for i in range(len(partition)):
                lower_bound_p = self._get_hoeffding_lower_bound(len_rewards_list[i] / sum_rewards_length,
                                                                self.confidence, sum_rewards_length)
                lower_bound_v = self._get_hoeffding_lower_bound(mean_best_rewards_list[i],
                                                                self.confidence, len_best_rewards_list[i],
                                                                max_best_rewards_list[i])
                partition_value = partition_value + lower_bound_p * lower_bound_v

            max_best_rewards_p_list.append(str(max_best_rewards_list))
            partition_values.append(partition_value)

        # Logging partition values
        # output_folder_path = get_resource_folder_path()+"/../"+"report/csv/partitions"
        # df_filepath = output_folder_path + "/part_values.csv"
        # try:
        #     if not os.path.exists(output_folder_path):
        #         os.mkdir(output_folder_path)
        # except FileNotFoundError:
        #     os.makedirs(output_folder_path)
        #
        # if os.path.exists(df_filepath):
        #     df = pd.read_csv(df_filepath, ":")
        # else:
        #     df = pd.DataFrame(columns=["v_"+str(p) for p in partitions] + ["m_v_"+str(p) for p in partitions])
        # df = df.append(pd.Series(partition_values + max_best_rewards_p_list, index=df.columns), ignore_index=True)
        # df.to_csv(df_filepath, ":", header=True, index=False)

        # Choose the partition with the highest value
        best_partition_idx = int(np.argmax(partition_values))
        return None, partitions[best_partition_idx]

    def _get_best_rewards_by_context_and_training_bandit(self, context: List[Tuple[int, int]]) -> List[float]:
        """
        Returns the best rewards related to the context defined by the list of selected features. The best rewards is
        the rewards related to the optimal arm and it is obtained by re-training offline the bandit with only the
        related information of that context and ask the bandit the optimal arm

        :param context: the list of min context
        :return: the list of best rewards related to the selected features and the index of the optimal arm
        """
        bandit: DiscreteBandit = self.bandit_class(**self.bandit_kwargs)
        rewards = self._get_rewards_per_context(context)
        pulled_arms = self._get_pulled_arms_per_context(context)

        for i in range(len(pulled_arms)):
            bandit.update(pulled_arms[i], rewards[i])
        best_rewards = bandit.rewards_per_arm[bandit.get_optimal_arm()]
        return best_rewards.copy()

    def _get_rewards_per_context(self, context: List[Tuple[int, int]]) -> List[float]:
        """
        Returns the rewards related to the context defined by the list of selected features

        :param context: the list of min_context
        :return: the list of rewards related to the selected features
        """
        rewards_per_selected_features = []
        for min_context in context:
            rewards_per_selected_features.extend(self.min_context_to_rewards_dict[min_context])
        return rewards_per_selected_features.copy()

    def _get_pulled_arms_per_context(self, context: List[Tuple[int, int]]) -> List[int]:
        """
        Return the pulled arms related to the context defined by the list of selected features

        :param context: the list of min_context
        :return: the list of pulled arms related to the selected features
        """
        pulled_arms_per_selected_features = []
        for min_context in context:
            pulled_arms_per_selected_features.extend(self.min_context_to_pulled_arm_dict[min_context])
        return pulled_arms_per_selected_features.copy()
