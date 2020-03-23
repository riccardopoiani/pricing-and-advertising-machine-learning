import unittest
import numpy as np

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer


class CampaignOptimizerTestCase(unittest.TestCase):
    def test_optimize_1(self):
        campaign = Campaign(2, 100, 50)
        campaign.set_sub_campaign(0, [3, 7, 14])
        campaign.set_sub_campaign(1, [2, 5, 7])

        opt_matrix, max_idx_matrix = CampaignOptimizer._optimize(campaign)

        true_opt_matrix = np.array([[0, 0, 0],
                                    [3, 7, 14],
                                    [5, 9, 16]])
        true_max_idx_matrix = np.array([[-1, -1, -1],
                                        [0, 1, 2],
                                        [0, 0, 0]])

        self.assertTrue((true_opt_matrix == opt_matrix).all())
        self.assertTrue((true_max_idx_matrix == max_idx_matrix).all())

    def test_optimize_2(self):
        campaign = Campaign(3, 90, 30)
        campaign.set_sub_campaign(0, [0, 3, 12, 20])
        campaign.set_sub_campaign(1, [0, 2, 7, 10])
        campaign.set_sub_campaign(2, [0, 5, 8, 12])

        opt_matrix, max_idx_matrix = CampaignOptimizer._optimize(campaign)

        true_opt_matrix = np.array([[0, 0, 0, 0],
                                    [0, 3, 12, 20],
                                    [0, 3, 12, 20],
                                    [0, 5, 12, 20]])
        true_max_idx_matrix = np.array([[-1, -1, -1, -1],
                                        [0, 1, 2, 3],
                                        [0, 0, 0, 0],
                                        [0, 1, 0, 0]])

        self.assertTrue((true_opt_matrix == opt_matrix).all())
        self.assertTrue((true_max_idx_matrix == max_idx_matrix).all())

    def test_optimize_3(self):
        campaign = Campaign(4, 100, 25)
        campaign.set_sub_campaign(0, [0, 3, 12, 20, 15])
        campaign.set_sub_campaign(1, [0, 2, 7, 10, 9])
        campaign.set_sub_campaign(2, [0, 5, 8, 12, 18])
        campaign.set_sub_campaign(3, [0, 9, 9, 10, 7])

        opt_matrix, max_idx_matrix = CampaignOptimizer._optimize(campaign)

        true_opt_matrix = np.array([[0, 0, 0, 0, 0],
                                    [0, 3, 12, 20, 20],
                                    [0, 3, 12, 20, 22],
                                    [0, 5, 12, 20, 25],
                                    [0, 9, 14, 21, 29]])
        true_max_idx_matrix = np.array([[-1, -1, -1, -1, -1],
                                        [0, 1, 2, 3, 3],
                                        [0, 0, 0, 0, 1],
                                        [0, 1, 0, 0, 1],
                                        [0, 1, 1, 1, 1]])

        self.assertTrue((true_opt_matrix == opt_matrix).all())
        self.assertTrue((true_max_idx_matrix == max_idx_matrix).all())

    def test_find_best_budgets_1(self):
        campaign = Campaign(2, 100, 50)
        campaign.set_sub_campaign(0, [3, 7, 14])
        campaign.set_sub_campaign(1, [2, 5, 7])

        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(campaign)

        self.assertEqual(max_clicks, 16)
        self.assertTrue((best_budgets == np.array([100, 0])).all())

    def test_find_best_budgets_2(self):
        campaign = Campaign(3, 90, 30)
        campaign.set_sub_campaign(0, [0, 3, 12, 20])
        campaign.set_sub_campaign(1, [0, 2, 7, 10])
        campaign.set_sub_campaign(2, [0, 5, 8, 12])

        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(campaign)

        self.assertEqual(max_clicks, 20)
        self.assertTrue((best_budgets == np.array([90, 0, 0])).all())

    def test_find_best_budgets_3(self):
        campaign = Campaign(4, 100, 25)
        campaign.set_sub_campaign(0, [0, 3, 12, 20, 15])
        campaign.set_sub_campaign(1, [0, 2, 7, 10, 9])
        campaign.set_sub_campaign(2, [0, 5, 8, 12, 18])
        campaign.set_sub_campaign(3, [0, 9, 9, 10, 7])

        max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(campaign)

        self.assertEqual(max_clicks, 29)
        self.assertTrue((best_budgets == np.array([75, 0, 0, 25])).all())


if __name__ == '__main__':
    unittest.main()
