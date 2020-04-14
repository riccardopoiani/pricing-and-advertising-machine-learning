import os
import sys
import numpy as np
import pandas as pd

from typing import List

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../../")

from environments.Settings.EnvironmentManager import EnvironmentManager
from environments.Settings.Scenario import Scenario
from utils.folder_management import handle_folder_creation

from utils.stats.StochasticFunction import IStochasticFunction

"""
Script for generating CSV files to create latex plots concerning scenarios
"""

SCENARIO_NAME = "linear_scenario"
FOLDER_RESULT = "../../report/csv/{}/".format(SCENARIO_NAME)
PLOT_INTERVAL = 3
PRICE_PLOT_N_POINTS = 100
ADS_PLOT_N_POINTS = 100
MIN_PRICE = 15
MAX_PRICE = 25
MIN_BUDGET = 0
MAX_DAILY_CUM_BUDGET = 1000

GET_PROFIT_DATA_DISAGGREGATED = True  # whether to get data regarding the profit of each class of users
GET_CRP_DATA_DISAGGREGATED = True  # whether to get data regarding the CRP of each class of users
GET_ADS_DATA_DISAGGREGATED = True  # whether to get data regarding the number of visit of each class of users

# Script begins
fd, folder_path_with_date = handle_folder_creation(result_path=FOLDER_RESULT, retrieve_text_file=False)
mean_scenario: Scenario = EnvironmentManager.load_scenario(SCENARIO_NAME,
                                                           get_mean_function=True)

# For all the phases print on a CSV data to reconstruct a function
for phase_idx, phase in enumerate(mean_scenario.get_phases()):

    crp_list: List[IStochasticFunction] = phase.get_crp_function()
    n_click_list: List[IStochasticFunction] = phase.get_n_clicks_function()

    # CRP data
    price_point_arr = np.linspace(MIN_PRICE, MAX_PRICE, PRICE_PLOT_N_POINTS)
    crp_data = np.zeros(shape=(len(crp_list) + 1, PRICE_PLOT_N_POINTS))
    crp_data[-1] = price_point_arr
    for i, crp in enumerate(crp_list):
        for j, point in enumerate(price_point_arr):
            crp_data[i][j] = crp.draw_sample(point)

    crp_df: pd.DataFrame = pd.DataFrame(crp_data.transpose())
    crp_df.rename(columns={0: "mean_crp_0", 1: "mean_crp_1", 2: "mean_crp_2", 3: "price"}, inplace=True)
    crp_df.to_csv("{}phase_{}_crp_data.csv".format(folder_path_with_date, phase_idx), index=False)

    # Profit data
    price_point_arr = np.linspace(MIN_PRICE, MAX_PRICE, PRICE_PLOT_N_POINTS)
    profit_data = np.zeros(shape=(len(crp_list) + 1, PRICE_PLOT_N_POINTS))
    profit_data[-1] = price_point_arr
    for i, crp in enumerate(crp_list):
        for j, point in enumerate(price_point_arr):
            profit_data[i][j] = crp.draw_sample(point)

    profit_df: pd.DataFrame = pd.DataFrame(crp_data.transpose())
    profit_df.rename(columns={0: "profit_0", 1: "profit_1", 2: "profit_2", 3: "price"}, inplace=True)
    profit_df.to_csv("{}phase_{}_profit_data.csv".format(folder_path_with_date, phase_idx), index=False)

    # Number of click data
    ads_point_arr = np.linspace(MIN_BUDGET, MAX_DAILY_CUM_BUDGET, ADS_PLOT_N_POINTS)
    ads_data = np.zeros(shape=(len(n_click_list) + 1, ADS_PLOT_N_POINTS))
    ads_data[-1] = ads_point_arr
    for i, n_click in enumerate(n_click_list):
        for j, point in enumerate(ads_point_arr):
            ads_data[i][j] = n_click.draw_sample(point)
    ads_df: pd.DataFrame = pd.DataFrame(ads_data.transpose())
    ads_df.rename(columns={0: "mean_click_0", 1: "mean_click_1", 2: "mean_click_2", 3: "budget"}, inplace=True)
    ads_df.to_csv("{}phase_{}_ads_data.csv".format(folder_path_with_date, phase_idx), index=False)
