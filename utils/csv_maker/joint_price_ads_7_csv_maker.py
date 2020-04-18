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

SCENARIO_NAME = "linear_scenario"
FOLDER_RESULT = "../../report/csv/point_7/{}/".format(SCENARIO_NAME)

N_ARMS_PRICE = 11
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

# Create scenario
mean_scenario: Scenario = EnvironmentManager.load_scenario(SCENARIO_NAME, get_mean_function=True)
crp_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_crp_function()
click_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_n_clicks_function()

# Compute optimal daily point (i.e you have to fix a price and under that constraint find all the things)
prices_arr = np.linspace(MIN_PRICE, MAX_PRICE, N_ARMS_PRICE)
best_profit_idx = 0
max_value = 0
best_budget_value = [0 for _ in range(mean_scenario.get_n_subcampaigns())]
campaign = Campaign(n_sub_campaigns=mean_scenario.get_n_subcampaigns(), cum_budget=DAILY_BUDGET, n_arms=N_ARMS_ADS)
for price_idx, price in enumerate(prices_arr):
    for i in range(campaign.get_n_sub_campaigns()):
        n_clicks = np.array([click_function_list[i].draw_sample(b)
                             for b in np.linspace(0, DAILY_BUDGET, N_ARMS_ADS)])
        click_value = crp_function_list[i].draw_sample(price) * (price - FIXED_COST)
        values: np.array = n_clicks * click_value
        campaign.set_sub_campaign_values(i, values)
    curr_value, curr_budgets = CampaignOptimizer.find_best_budgets(campaign)

    if curr_value > max_value:
        best_profit_idx = price_idx
        max_value = curr_value
        best_budget_value = curr_budgets

print("Best budget allocation is reached for price = {}\n".format(prices_arr[best_profit_idx]))
print("Best budget allocation is {}\n".format(best_budget_value))
print("The expected best daily profit is given by {}\n".format(max_value))

# Instantaneous rewards computation
if CSV_REWARD:
    n_days = len(total_reward_list[0][0])
    mean_reward = np.zeros(shape=(n_bandit+1, n_days))
    std_reward = np.zeros(shape=(n_bandit+1, n_days))
    mean_reward[-1] = np.arange(n_days)
    std_reward[-1] = np.arange(n_days)

    for bandit_idx in range(n_bandit):
        n_exp = len(total_reward_list[bandit_idx])

        for curr_day in range(n_days):
            daily_values = []
            for exp in range(n_exp):
                daily_values.append(total_reward_list[bandit_idx][exp][curr_day])

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
if CSV_REGRET:
    n_days = len(total_reward_list[0][0])
    mean_data = np.zeros(shape=(n_bandit + 1, n_days))
    std_data = np.zeros(shape=(n_bandit + 1, n_days))
    mean_data[-1] = np.arange(n_days)
    std_data[-1] = np.arange(n_days)

    for bandit_idx in range(n_bandit):
        n_exp = len(total_reward_list[bandit_idx])
        values = [[] for _ in range(n_days)]
        for exp in range(n_exp):
            curr_exp_value = 0
            for curr_day in range(n_days):
                curr_exp_value += total_reward_list[bandit_idx][exp][curr_day]
                values[curr_day].append((curr_day + 1) * max_value - curr_exp_value)
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
    total_df.to_csv("{}regret.csv".format(folder_path_with_date), index=False)
