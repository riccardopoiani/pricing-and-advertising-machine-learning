import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd

from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../../")

from environments.Settings.EnvironmentManager import EnvironmentManager
from environments.Settings.Scenario import Scenario
from utils.folder_management import handle_folder_creation
from utils.stats.StochasticFunction import IStochasticFunction

FOLDER_RESULT = "../../report/csv/advertising_2/"

CSV_CUM_REGRET = True

SCENARIO_NAME = "linear_scenario"
N_ARMS_ADV = 11
BUDGET = 1000
PRICE_PLOT_N_POINTS = 100
ADS_PLOT_N_POINTS = 100
MIN_PRICE = 15
MAX_PRICE = 25
FIXED_COST = 12
REWARD_FILE_LIST = ["../../report/project_point_2/Apr17_17-09-39/reward_GPBandit.pkl",
                    "../../report/project_point_2/Apr24_14-15-39/reward_GaussianBandit.pkl"]

#BANDIT_NAME = ["GPBandit"]
BANDIT_NAME = ["GPBandit", "GaussianBandit"]

n_bandit = len(BANDIT_NAME)
_, folder_path_with_date = handle_folder_creation(result_path=FOLDER_RESULT, retrieve_text_file=False)

assert len(REWARD_FILE_LIST) == len(BANDIT_NAME), "Number of bandits and file list does not match"

# Reading file list
total_reward_list = []

for curr_day, _ in enumerate(BANDIT_NAME):
    rewards = []
    with (open(REWARD_FILE_LIST[curr_day], "rb")) as openfile:
        while True:
            try:
                rewards.append(pickle.load(openfile))
            except EOFError:
                break

    rewards = rewards[0]
    total_reward_list.append(rewards)

# Compute N-days
n_days = len(total_reward_list[0][0][0])

# Compute mean and standard deviation for each day
mean_reward = np.zeros(shape=(n_bandit + 1, n_days))
std_reward = np.zeros(shape=(n_bandit + 1, n_days))
mean_reward[-1] = np.arange(n_days) + 1
std_reward[-1] = np.arange(n_days) + 1

for bandit_idx in range(len(BANDIT_NAME)):
    n_exp = len(total_reward_list[bandit_idx])

    for curr_day in range(n_days):
        daily_values = []
        for exp in range(n_exp):
            daily_values.append(total_reward_list[bandit_idx][exp][0][curr_day])

        mean_reward[bandit_idx][curr_day] = np.array(daily_values).mean()
        std_reward[bandit_idx][curr_day] = np.array(daily_values).std()

mean_df = pd.DataFrame(mean_reward.transpose())
std_df = pd.DataFrame(std_reward.transpose())

for bandit_idx, name in enumerate(BANDIT_NAME):
    mean_df.rename(columns={bandit_idx: "mean_reward_{}".format(name)}, inplace=True)

mean_df.rename(columns={n_bandit: "day"}, inplace=True)

for bandit_idx, name in enumerate(BANDIT_NAME):
    std_df.rename(columns={bandit_idx: "mean_std_{}".format(name)}, inplace=True)

std_df.rename(columns={n_bandit: "day"}, inplace=True)

total_df = mean_df.merge(std_df, left_on="day", right_on="day")
total_df.to_csv("{}instant_reward.csv".format(folder_path_with_date), index=False)

# Something
mean_scenario: Scenario = EnvironmentManager.load_scenario(SCENARIO_NAME, get_mean_function=True)
click_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_n_clicks_function()

# Optimal point computation
campaign = Campaign(n_sub_campaigns=mean_scenario.get_n_subcampaigns(), cum_budget=BUDGET, n_arms=N_ARMS_ADV)
for i in range(campaign.get_n_sub_campaigns()):
    sub_campaign_values = [click_function_list[i].draw_sample(b) for b in np.linspace(0, BUDGET, N_ARMS_ADV)]
    campaign.set_sub_campaign_values(i, sub_campaign_values)
max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(campaign)

# # Compute regret
# if CSV_CUM_REGRET:
#     mean_regret_data = np.zeros(shape=(n_bandit + 1, n_days))
#     std_regret_data = np.zeros(shape=(n_bandit + 1, n_days))
#     mean_regret_data[-1] = np.arange(n_days) + 1
#     std_regret_data[-1] = np.arange(n_days) + 1
#
#     for bandit_idx in range(len(BANDIT_NAME)):
#         n_exp = len(total_reward_list[bandit_idx])
#
#         for curr_day in range(n_days):
#             daily_values = []
#             for exp in range(n_exp):
#                 daily_values.append(max_clicks - total_reward_list[bandit_idx][exp][0][curr_day])
#
#             mean_regret_data[bandit_idx][curr_day] = np.array(daily_values).mean()
#             std_regret_data[bandit_idx][curr_day] = np.array(daily_values).std()
#
#     mean_df = pd.DataFrame(np.cumsum(mean_regret_data).transpose())
#     std_df = pd.DataFrame(std_regret_data.transpose())
#
#     for bandit_idx, name in enumerate(BANDIT_NAME):
#         mean_df.rename(columns={bandit_idx: "mean_regret_{}".format(name)}, inplace=True)
#
#     mean_df.rename(columns={n_bandit: "day"}, inplace=True)
#     for bandit_idx, name in enumerate(BANDIT_NAME):
#         std_df.rename(columns={bandit_idx: "std_regret_{}".format(name)}, inplace=True)
#     std_df.rename(columns={n_bandit: "day"}, inplace=True)
#
#     total_df = mean_df.merge(std_df, left_on="day", right_on="day")
#     total_df.to_csv("{}daily_discrete_regret.csv".format(folder_path_with_date), index=False)

if CSV_CUM_REGRET:
    mean_data = np.zeros(shape=(n_bandit + 1, n_days))
    std_data = np.zeros(shape=(n_bandit + 1, n_days))
    mean_data[-1] = np.arange(n_days)
    std_data[-1] = np.arange(n_days)

    for bandit_idx in range(n_bandit):
        n_exp = len(total_reward_list[bandit_idx])
        values = [[] for _ in range(n_days)]
        for exp in range(n_exp):
            curr_exp_value = 0
            for day in range(n_days):
                curr_exp_value += total_reward_list[bandit_idx][exp][0][day]
                values[day].append((day + 1) * max_clicks - curr_exp_value)
        for day in range(n_days):
            mean_data[bandit_idx][day] = np.array(values[day]).mean()
            std_data[bandit_idx][day] = np.array(values[day]).std()

    mean_df = pd.DataFrame(mean_data.transpose())
    std_df = pd.DataFrame(std_data.transpose())
    for bandit_idx, name in enumerate(BANDIT_NAME):
        mean_df.rename(columns={bandit_idx: "mean_regret_{}".format(name)}, inplace=True)

    mean_df.rename(columns={n_bandit: "day"}, inplace=True)
    for bandit_idx, name in enumerate(BANDIT_NAME):
        std_df.rename(columns={bandit_idx: "std_regret_{}".format(name)}, inplace=True)
    std_df.rename(columns={n_bandit: "day"}, inplace=True)
    total_df = mean_df.merge(std_df, left_on="day", right_on="day")
    total_df.to_csv("{}discrete_cum_regret.csv".format(folder_path_with_date), index=False)