import os
import pickle
import sys
from typing import List

import pandas as pd
import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../../")

from utils.folder_management import handle_folder_creation
from advertising.data_structure.Campaign import Campaign
from advertising.optimizers.CampaignOptimizer import CampaignOptimizer
from environments.Settings.EnvironmentManager import EnvironmentManager
from environments.Settings.Scenario import Scenario
from utils.stats.StochasticFunction import IStochasticFunction

CSV_REWARD = True
CSV_REGRET = True

SCENARIO_NAME = "non_stationary"
FOLDER_RESULT = "../../report/csv/point_3/{}/".format(SCENARIO_NAME)

N_ARMS_ADS = 11
DAILY_BUDGET = 1000
MIN_PRICE = 15
MAX_PRICE = 25
FIXED_COST = 12

REWARD_FILE_LIST = ["../../report/project_point_6_7/Apr18_11-17-53/total_reward_JBFTS.pkl",
                    "../../report/project_point_6_7/Apr18_14-41-22/total_reward_JBBQ.pkl",
                    "../../report/project_point_6_7/Apr18_16-35-04/total_reward_JBBExp.pkl",
                    "../../report/project_point_6_7/Apr18_17-10-06/total_reward_JBBQ.pkl",
                    "../../report/project_point_6_7/Apr18_17-35-16/total_reward_JBBExp.pkl"]

BANDIT_NAME = ["JBFTS", "JBBQ", "JBBExp", "JBBQD", "JBBExpD"]

n_bandit = len(BANDIT_NAME)
_, folder_path_with_date = handle_folder_creation(result_path=FOLDER_RESULT, retrieve_text_file=False)

assert len(REWARD_FILE_LIST) == len(BANDIT_NAME), "Number of bandits and file list does not match"

# Data reading
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

# Optimal point computation
mean_scenario: Scenario = EnvironmentManager.load_scenario(SCENARIO_NAME, get_mean_function=True)
phase_len_list = [p.get_duration() for p in mean_scenario.get_phases()]
optimal_reward_per_phase = [0 for _ in range(len(phase_len_list))]
for phase_idx, phase in enumerate(mean_scenario.get_phases()):
    click_functions: List[IStochasticFunction] = phase.get_n_clicks_function()
    campaign = Campaign(n_sub_campaigns=mean_scenario.get_n_subcampaigns(), cum_budget=DAILY_BUDGET, n_arms=N_ARMS_ADS)
    for i in range(campaign.get_n_sub_campaigns()):
        sub_campaign_values = [click_functions[i].draw_sample(b) for b in np.linspace(0, DAILY_BUDGET, N_ARMS_ADS)]
        campaign.set_sub_campaign_values(i, sub_campaign_values)
    max_clicks, best_budgets = CampaignOptimizer.find_best_budgets(campaign)
    print("Phase {} maximum number of clicks: {}".format(phase_idx, max_clicks))
    print("Phase {} budget allocation: {}".format(phase_idx, best_budgets))

    optimal_reward_per_phase[phase_idx] = max_clicks

n_days = len(total_reward_list[0][0][0])

# Instant reward computation
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

# Regret computation
mean_data = np.zeros(shape=(n_bandit + 1, n_days))
std_data = np.zeros(shape=(n_bandit + 1, n_days))
mean_data[-1] = np.arange(n_days)
std_data[-1] = np.arange(n_days)

for bandit_idx in range(n_bandit):
    n_exp = len(total_reward_list[bandit_idx])
    values = [[] for _ in range(n_days)]
    for exp in range(n_exp):
        curr_exp_value = 0
        curr_best_value = 0
        for day in range(n_days):
            if day <= phase_len_list[0]:
                best_value = optimal_reward_per_phase[0]
            elif day <= phase_len_list[1]:
                best_value = optimal_reward_per_phase[1]
            else:
                best_value = optimal_reward_per_phase[2]
            curr_best_value += best_value

            curr_exp_value += total_reward_list[bandit_idx][exp][0][day]
            values[day].append(curr_best_value - curr_exp_value)
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
