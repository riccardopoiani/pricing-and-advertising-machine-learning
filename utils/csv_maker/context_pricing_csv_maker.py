import os
import sys
import pickle
from typing import List

import numpy as np
import pandas as pd

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../../")

from environments.Settings.EnvironmentManager import EnvironmentManager
from environments.Settings.Scenario import Scenario
from utils.folder_management import handle_folder_creation
from utils.stats.StochasticFunction import IStochasticFunction


SCENARIO_NAME = "linear_visit_tanh_price"
FOLDER_RESULT = "../../report/csv/point_5/{}/".format(SCENARIO_NAME)

CSV_DAILY_DISCRETE_REGRET = True

N_ARMS_PRICE = 11
FIXED_BUDGET = [1000. / 3, 1000. / 3, 1000. / 3]
MIN_PRICE = 15
MAX_PRICE = 25
FIXED_COST = 12

REWARD_FILE_LIST = ["../../report/project_point_5/Apr21_19-01-23/reward_TS.pkl",
                    "../../report/project_point_5/Apr24_19-08-45/reward_TS.pkl",
                    "../../report/project_point_5/Apr24_20-06-16/reward_TS.pkl"]

DAYS_FILE_LIST = ["../../report/project_point_5/Apr21_19-01-23/day_TS.pkl",
                  "../../report/project_point_5/Apr24_19-08-45/day_TS.pkl",
                  "../../report/project_point_5/Apr24_20-06-16/day_TS.pkl"]

BANDIT_NAME = ["CONF_0dot1", "CONF_0dot05", "CONF_0dot01"]


n_bandit = len(BANDIT_NAME)
_, folder_path_with_date = handle_folder_creation(result_path=FOLDER_RESULT, retrieve_text_file=False)

assert len(REWARD_FILE_LIST) == len(BANDIT_NAME), "Number of bandits and file list does not match"
assert len(REWARD_FILE_LIST) == len(DAYS_FILE_LIST), "Number of bandits and file list does not match"

# Reading file list
total_reward_list = []
total_day_list = []

for curr_day, _ in enumerate(BANDIT_NAME):
    rewards = []
    with (open(REWARD_FILE_LIST[curr_day], "rb")) as openfile:
        while True:
            try:
                rewards.append(pickle.load(openfile))
            except EOFError:
                break

    days = []
    with (open(DAYS_FILE_LIST[curr_day], "rb")) as openfile:
        while True:
            try:
                days.append(pickle.load(openfile))
            except EOFError:
                break
    rewards = rewards[0]
    days = days[0]
    total_reward_list.append(rewards)
    total_day_list.append(days)

# Compute N-days
n_days = np.inf
for curr_day, day_list_bandit in enumerate(total_day_list):
    for j, day_list_exp in enumerate(day_list_bandit):
        if len(day_list_exp) < n_days:
            n_days = len(day_list_exp)
n_days = n_days - 1

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
            start_user = total_day_list[bandit_idx][exp][curr_day]
            end_user = total_day_list[bandit_idx][exp][curr_day + 1]
            daily_values.append(np.array(total_reward_list[bandit_idx][exp][start_user:end_user]).sum())

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
total_df.to_csv("{}reward.csv".format(folder_path_with_date), index=False)

# Aggregated plot under a fixed budget
mean_scenario: Scenario = EnvironmentManager.load_scenario(SCENARIO_NAME, get_mean_function=True)
crp_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_crp_function()
click_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_n_clicks_function()

# Regret computation with correct arms (and not all the function)
optimal_discrete_profits = []
optimal_arms = []
prices = np.linspace(start=MIN_PRICE, stop=MAX_PRICE, num=N_ARMS_PRICE)
for i, crp in enumerate(crp_function_list):
    crp_values = np.array([crp.draw_sample(x) for x in prices])
    profit_values = crp_values * (prices - FIXED_COST)
    optimal_discrete_profits.append(profit_values.max())
    optimal_arms.append(prices[profit_values.argmax()])

daily_n_clicks = np.array([f.draw_sample(FIXED_BUDGET[i]) for i, f in enumerate(click_function_list)])
optimal_mean_daily_reward_discrete = sum(optimal_discrete_profits * daily_n_clicks)

print("Optimal prices for each subcampaign is {}\n".format(optimal_arms))
print("Optimal mean daily reward is {}".format(optimal_mean_daily_reward_discrete))

if CSV_DAILY_DISCRETE_REGRET:
    mean_regret_data = np.zeros(shape=(n_bandit + 1, n_days))
    std_regret_data = np.zeros(shape=(n_bandit + 1, n_days))
    mean_regret_data[-1] = np.arange(n_days) + 1
    std_regret_data[-1] = np.arange(n_days) + 1

    for bandit_idx in range(len(BANDIT_NAME)):
        n_exp = len(total_reward_list[bandit_idx])

        for curr_day in range(n_days):
            daily_values = []
            for exp in range(n_exp):
                end_user = total_day_list[bandit_idx][exp][curr_day + 1]
                daily_values.append((curr_day + 1) * optimal_mean_daily_reward_discrete - np.array(
                    total_reward_list[bandit_idx][exp][0:end_user]).sum())

            mean_regret_data[bandit_idx][curr_day] = np.array(daily_values).mean()
            std_regret_data[bandit_idx][curr_day] = np.array(daily_values).std()

    mean_df = pd.DataFrame(mean_regret_data.transpose())
    std_df = pd.DataFrame(std_regret_data.transpose())
    for bandit_idx, name in enumerate(BANDIT_NAME):
        mean_df.rename(columns={bandit_idx: "mean_regret_{}".format(name)}, inplace=True)

    mean_df.rename(columns={n_bandit: "day"}, inplace=True)
    for bandit_idx, name in enumerate(BANDIT_NAME):
        std_df.rename(columns={bandit_idx: "std_regret_{}".format(name)}, inplace=True)
    std_df.rename(columns={n_bandit: "day"}, inplace=True)
    total_df = mean_df.merge(std_df, left_on="day", right_on="day")
    total_df.to_csv("{}regret.csv".format(folder_path_with_date), index=False)
