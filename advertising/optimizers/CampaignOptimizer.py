from bandit.discrete.GPBandit import GPBandit
from advertising.model.Campaign import Campaign
import numpy as np


class CampaignOptimizer(object):

    def __init__(self, campaign: Campaign, init_std_dev):
        self._campaign = campaign
        self._n_clicks_estimators = [GPBandit(campaign.get_budgets(), init_std_dev)
                                     for _ in range(campaign.get_n_sub_campaigns())]

    def round(self):
        """

        :return:
        """
        for sub_campaign_idx, estimator in enumerate(self._n_clicks_estimators):
            sub_campaign_values = estimator.sample()
            self._campaign.set_sub_campaign(sub_campaign_idx, sub_campaign_values)

        optimization_matrix, max_idx_matrix = self.optimize()
        return self.find_best_budgets(optimization_matrix, max_idx_matrix)

    def optimize(self):
        """

        :return:
        """
        optimization_matrix = np.zeros(shape=(self._campaign.get_n_sub_campaigns() + 1, len(self._campaign.get_budgets())))
        max_idx_matrix = np.full_like(optimization_matrix, fill_value=-1, dtype=np.int)
        prev_row = 0
        # cum_budget = self._campaign.get_cum_budget()
        clicks_matrix = self._campaign.get_sub_campaigns()
        for row in range(1, optimization_matrix.shape[0]):
            temp_clicks = clicks_matrix[row - 1][::-1]

            for col in range(optimization_matrix.shape[1]):
                cum_sum_clicks = temp_clicks[optimization_matrix.shape[1] - col - 1:] + optimization_matrix[prev_row, :col + 1]
                idx_max = np.argmax(cum_sum_clicks)

                optimization_matrix[row, col] = cum_sum_clicks[idx_max]
                max_idx_matrix[row, col] = col - idx_max
            prev_row = row
        return optimization_matrix, max_idx_matrix

    def find_best_budgets(self, optimization_matrix, max_idx_matrix):
        max_clicks_idx = np.argmax(optimization_matrix[-1])
        max_clicks = optimization_matrix[-1, max_clicks_idx]

        best_budgets = []
        remaining_budget = self._campaign.get_budgets()[max_clicks_idx]
        for i in range(optimization_matrix.shape[0] - 1, 0, -1):
            curr_row_budget = self._campaign.get_budgets()[max_idx_matrix[i, max_clicks_idx]]
            best_budgets.append(curr_row_budget)
            remaining_budget -= curr_row_budget
            max_clicks_idx = np.where(self._campaign.get_budgets() == remaining_budget)[0]
        best_budgets = best_budgets[::-1]

        return max_clicks, best_budgets

