from typing import List

from advertising.data_structure.Campaign import Campaign
import numpy as np


class CampaignOptimizer(object):
    """
    Helper class that contains all the function used to optimize the combinatorial problem, i.e. campaign advertising
    problem
    """

    @classmethod
    def _optimize(cls, campaign: Campaign) -> (np.ndarray, np.ndarray):
        """
        Optimize the combinatorial problem of the advertising campaign by using a dynamic programming algorithm

        :param campaign: the campaign to be optimized
        :return:
            - the optimization matrix (N+1) x M containing, for each pair (budget, set of sub-campaign), the maximum number
              of clicks achieavable
            - the maximum indices (N+1) x M related to the optimization matrix containing, for each pair
              (budget, set of sub-campaign), the index of the best budget for the new added sub-campaign w.r.t. previous
              set of sub-campaign (i.e. row)
            where N is the number of sub-campaign and M is the number of discrete budgets
        """
        optimization_matrix = np.zeros(shape=(campaign.get_n_sub_campaigns() + 1, len(campaign.get_budgets())))
        max_idx_matrix = np.full_like(optimization_matrix, fill_value=-1, dtype=np.int)
        prev_row = 0
        # cum_budget = self._campaign.get_cum_budget()
        clicks_matrix = campaign.get_sub_campaigns()
        for row in range(1, optimization_matrix.shape[0]):
            temp_clicks = clicks_matrix[row - 1][::-1]

            for col in range(optimization_matrix.shape[1]):
                cum_sum_clicks = temp_clicks[optimization_matrix.shape[1] - col - 1:] + optimization_matrix[prev_row, :col + 1]
                idx_max = np.argmax(cum_sum_clicks)

                optimization_matrix[row, col] = cum_sum_clicks[idx_max]
                max_idx_matrix[row, col] = col - idx_max
            prev_row = row
        return optimization_matrix, max_idx_matrix

    @classmethod
    def find_best_budgets(cls, campaign) -> (float, List[float]):
        """
        Find, for the campaign, the best allocation of budgets for each sub-campaign by using the DP algorithm and by
        exploiting the max_idx_matrix 'recursively'

        :param campaign: the campaign of which you want to find the best allocation of budgets
        :return:
            - maximum number of clicks achieavable with the best allocation of budgets
            - best allocation of budgets for each sub-campaign
        """
        optimization_matrix, max_idx_matrix = cls._optimize(campaign)
        max_clicks_idx = np.argmax(optimization_matrix[-1])
        max_clicks = optimization_matrix[-1, max_clicks_idx]

        best_budgets = []
        remaining_budget = campaign.get_budgets()[max_clicks_idx]
        for i in range(optimization_matrix.shape[0] - 1, 0, -1):
            curr_row_budget = campaign.get_budgets()[max_idx_matrix[i, max_clicks_idx]]
            best_budgets.append(curr_row_budget)
            remaining_budget -= curr_row_budget
            max_clicks_idx = np.where(campaign.get_budgets() == remaining_budget)[0][0]
        best_budgets = best_budgets[::-1]

        return max_clicks, best_budgets

