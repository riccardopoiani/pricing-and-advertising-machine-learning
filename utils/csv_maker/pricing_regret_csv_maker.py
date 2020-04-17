import os
import sys
import pickle
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../../")

from environments.Settings.EnvironmentManager import EnvironmentManager
from environments.Settings.Scenario import Scenario
from utils.folder_management import handle_folder_creation
from utils.stats.StochasticFunction import IStochasticFunction, AggregatedFunction, MultipliedStochasticFunction

FOLDER_RESULT = "../../report/csv/pricing_bandit/"

CSV_DISCRETE_USER_REGRET = True
CSV_CONTINUE_USER_REGRET = True
CSV_DAILY_DISCRETE_REGRET = True
CSV_DAILY_CONT_REGRET = True

SCENARIO_NAME = "linear_scenario"
N_ARMS_PRICE = 10
FIXED_BUDGET = [1000 / 3, 1000 / 3, 1000 / 3]
PRICE_PLOT_N_POINTS = 100
ADS_PLOT_N_POINTS = 100
MIN_PRICE = 15
MAX_PRICE = 25
FIXED_COST = 12
REWARD_FILE_LIST = ["../../report/project_point_4/Apr16_22-58-37/reward_TS.pkl",
                    "../../report/project_point_4/Apr16_23-06-28/reward_UCB1.pkl",
                    "../../report/project_point_4/Apr16_23-07-18/reward_UCBL.pkl",
                    "../../report/project_point_4/Apr16_23-10-28/reward_UCB1M.pkl",
                    "../../report/project_point_4/Apr16_23-14-22/reward_UCBLM.pkl",
                    "../../report/project_point_4/Apr16_23-15-33/reward_EXP3.pkl"]

DAYS_FILE_LIST = ["../../report/project_point_4/Apr16_22-58-37/day_TS.pkl",
                  "../../report/project_point_4/Apr16_23-06-28/day_UCB1.pkl",
                  "../../report/project_point_4/Apr16_23-07-18/day_UCBL.pkl",
                  "../../report/project_point_4/Apr16_23-10-28/day_UCB1M.pkl",
                  "../../report/project_point_4/Apr16_23-14-22/day_UCBLM.pkl",
                  "../../report/project_point_4/Apr16_23-15-33/day_EXP3.pkl"]

BANDIT_NAME = ["TS", "UCB1", "UCBL", "UCB1M", "UCBLM", "EXP3"]

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
total_df.to_csv("{}instant_reward.csv".format(folder_path_with_date), index=False)

# Aggregated plot under a fixed budget
mean_scenario: Scenario = EnvironmentManager.load_scenario(SCENARIO_NAME, get_mean_function=True)
crp_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_crp_function()
click_function_list: List[IStochasticFunction] = mean_scenario.get_phases()[0].get_n_clicks_function()

context_weight = np.array([f.draw_sample(FIXED_BUDGET[i]) for i, f in enumerate(click_function_list)])
context_weight = context_weight / context_weight.sum()  # weight to to retrieve the aggregated CRP

aggregated_crp: AggregatedFunction = AggregatedFunction(f_list=crp_function_list, weights=context_weight)

price_point_arr = np.linspace(MIN_PRICE, MAX_PRICE, PRICE_PLOT_N_POINTS)
crp_data = np.zeros(shape=(1 + 1, PRICE_PLOT_N_POINTS))
crp_data[-1] = price_point_arr
for j, point in enumerate(price_point_arr):
    crp_data[0][j] = aggregated_crp.draw_sample(point)

crp_df: pd.DataFrame = pd.DataFrame(crp_data.transpose())
crp_df.rename(columns={0: "mean_aggr_crp", 1: "price"}, inplace=True)
crp_df.to_csv("{}aggregated_crp_data.csv".format(folder_path_with_date), index=False)

price_point_arr = np.linspace(MIN_PRICE, MAX_PRICE, PRICE_PLOT_N_POINTS)
profit_data = np.zeros(shape=(1 + 1, PRICE_PLOT_N_POINTS))
profit_data[-1] = price_point_arr
for j, point in enumerate(price_point_arr):
    profit_data[0][j] = aggregated_crp.draw_sample(point) * (point - 12)

profit_df: pd.DataFrame = pd.DataFrame(profit_data.transpose())
profit_df.rename(columns={0: "profit_0", 1: "price"}, inplace=True)
profit_df.to_csv("{}aggregated_profit_data.csv".format(folder_path_with_date), index=False)

# Optimal point computation
aggregated_profit: MultipliedStochasticFunction = MultipliedStochasticFunction(aggregated_crp, shift=-FIXED_COST)
min_result = minimize_scalar(aggregated_profit.get_minus_lambda(), bounds=(MIN_PRICE, MAX_PRICE), method="bounded")
optimal_mean_reward_user = aggregated_profit.draw_sample(min_result["x"])
average_daily_users = np.array([f.draw_sample(FIXED_BUDGET[i]) for i, f in enumerate(click_function_list)]).sum()
optimal_mean_daily_reward = optimal_mean_reward_user * average_daily_users
print("Optimal mean reward is {}, reached at x={}\n".format(optimal_mean_reward_user, min_result["x"]))
print("Optimal mean daily reward is {}, since there are {} daily users".format(optimal_mean_daily_reward,
                                                                               average_daily_users))
# Compute regret
if CSV_DAILY_CONT_REGRET:
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
                daily_values.append((curr_day + 1) * optimal_mean_daily_reward - np.array(
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
    total_df.to_csv("{}daily_regret.csv".format(folder_path_with_date), index=False)

# Regret computation with correct arms (and not all the function)
aggregated_profit: MultipliedStochasticFunction = MultipliedStochasticFunction(aggregated_crp, shift=-FIXED_COST)
points = np.linspace(start=MIN_PRICE, stop=MAX_PRICE, num=N_ARMS_PRICE)
profit_points = np.array([aggregated_profit.draw_sample(x) for x in points])
optimal_discrete_profit = profit_points.max()
average_daily_users = np.array([f.draw_sample(FIXED_BUDGET[i]) for i, f in enumerate(click_function_list)]).sum()
optimal_mean_daily_reward_discrete = optimal_discrete_profit * average_daily_users

print("Optimal mean discrete reward is {}, reached for arm index = {}\n".format(optimal_mean_daily_reward_discrete,
                                                                                profit_points.argmax()))
print("Optimal mean daily reward is {}, since there are {} daily users".format(optimal_mean_daily_reward,
                                                                               average_daily_users))

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
    total_df.to_csv("{}daily_discrete_regret.csv".format(folder_path_with_date), index=False)

# Compute regret user-wise
if CSV_DISCRETE_USER_REGRET:
    total_users = len(total_reward_list[0][0])
    mean_data = np.zeros(shape=(n_bandit + 1, total_users))
    std_data = np.zeros(shape=(n_bandit + 1, total_users))
    mean_data[-1] = np.arange(total_users)
    std_data[-1] = np.arange(total_users)

    for bandit_idx in range(n_bandit):
        n_exp = len(total_reward_list[bandit_idx])
        values = [[] for _ in range(total_users)]
        for exp in range(n_exp):
            curr_exp_value = 0
            for user in range(total_users):
                curr_exp_value += total_reward_list[bandit_idx][exp][user]
                values[user].append((user + 1) * optimal_discrete_profit - curr_exp_value)
        for user in range(total_users):
            mean_data[bandit_idx][user] = np.array(values[user]).mean()
            std_data[bandit_idx][user] = np.array(values[user]).std()

    mean_df = pd.DataFrame(mean_data.transpose())
    std_df = pd.DataFrame(std_data.transpose())
    for bandit_idx, name in enumerate(BANDIT_NAME):
        mean_df.rename(columns={bandit_idx: "mean_regret_{}".format(name)}, inplace=True)

    mean_df.rename(columns={n_bandit: "user"}, inplace=True)
    for bandit_idx, name in enumerate(BANDIT_NAME):
        std_df.rename(columns={bandit_idx: "std_regret_{}".format(name)}, inplace=True)
    std_df.rename(columns={n_bandit: "user"}, inplace=True)
    total_df = mean_df.merge(std_df, left_on="user", right_on="user")
    total_df.to_csv("{}discrete_user_regret.csv".format(folder_path_with_date), index=False)

# Compute regret user-wise with real loss
if CSV_CONTINUE_USER_REGRET:
    total_users = len(total_reward_list[0][0])
    mean_data = np.zeros(shape=(n_bandit+1, total_users))
    std_data = np.zeros(shape=(n_bandit+1, total_users))
    mean_data[-1] = np.arange(total_users)
    std_data[-1] = np.arange(total_users)

    for bandit_idx in range(n_bandit):
        n_exp = len(total_reward_list[bandit_idx])
        values = [[] for _ in range(total_users)]
        for exp in range(n_exp):
            curr_exp_value = 0
            for user in range(total_users):
                curr_exp_value += total_reward_list[bandit_idx][exp][user]
                values[user].append((user + 1) * optimal_mean_reward_user - curr_exp_value)
        for user in range(total_users):
            mean_data[bandit_idx][user] = np.array(values[user]).mean()
            std_data[bandit_idx][user] = np.array(values[user]).std()

    mean_df = pd.DataFrame(mean_data.transpose())
    std_df = pd.DataFrame(std_data.transpose())
    for bandit_idx, name in enumerate(BANDIT_NAME):
        mean_df.rename(columns={bandit_idx: "mean_regret_{}".format(name)}, inplace=True)

    mean_df.rename(columns={n_bandit: "user"}, inplace=True)
    for bandit_idx, name in enumerate(BANDIT_NAME):
        std_df.rename(columns={bandit_idx: "std_regret_{}".format(name)}, inplace=True)
    std_df.rename(columns={n_bandit: "user"}, inplace=True)
    total_df = mean_df.merge(std_df, left_on="user", right_on="user")
    total_df.to_csv("{}cont_user_regret.csv".format(folder_path_with_date), index=False)
