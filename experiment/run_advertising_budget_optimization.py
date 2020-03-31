import argparse
import os
import pickle
import sys
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from advertising.regressors import DiscreteRegressor
from advertising.regressors.DiscreteGPRegressor import DiscreteGPRegressor
from advertising.regressors.DiscreteGaussianRegressor import DiscreteGaussianRegressor
from advertising.data_structure.Campaign import Campaign
from bandit.combinatiorial.CombinatorialStationaryBandit import CombinatorialStationaryBandit
from bandit.combinatiorial.SWCombinatorialBandit import SWCombinatorialBandit
from bandit.combinatiorial.CDCombinatorialBandit import CDCombinatorialBandit
from environments.AdvertisingEnvironment import AdvertisingEnvironment
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation

# Basic default settings
N_ROUNDS = 100
BASIC_OUTPUT_FOLDER = "../report/project_point_2/"

# Bandit parameters
ALPHA = 100
N_RESTARTS_OPTIMIZERS = 10

# Advertising settings
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"
CUM_BUDGET = 10000
N_ARMS = 11
BEST_BUDGET_ALLOCATION = [3000, 2000, 5000]


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()

    # Ads setting
    parser.add_argument("-bud", "--cum_budget", default=CUM_BUDGET,
                        help="Cumulative budget to be used for the simulation",
                        type=float)
    parser.add_argument("-n_arms", "--n_arms", help="Number of arms for the prices", type=int, default=N_ARMS)

    # Scenario settings
    parser.add_argument("-scenario_name", "--scenario_name", default=SCENARIO_NAME,
                        type=str, help="Name of the setting of the experiment")

    # Experiment run details
    parser.add_argument("-n_runs", "--n_runs", default=1, help="Number of runs of the experiments", type=int)
    parser.add_argument("-n_jobs", "--n_jobs", default=1, help="Number of jobs to be run", type=int)
    parser.add_argument("-t", "--n_rounds", default=N_ROUNDS, help="Number of rounds", type=int)

    # Bandit hyper-parameters
    parser.add_argument("-b", "--bandit_name", help="Name of the bandit to be used in the experiment")
    parser.add_argument("-sw", "--sw_size", help="Size of the sliding window for SW bandits", type=int)
    parser.add_argument("-gamma", "--gamma",
                        help="Controls the fraction of the uniform sampling over the number of arms",
                        type=int, default=0.1)
    parser.add_argument("-cd_threshold", "--cd_threshold", help="Threshold used for change detection",
                        type=int, default=0.1)

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=1)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)

    return parser.parse_args()


def get_bandit(bandit_name: str, campaign: Campaign, init_std_dev: float = 1e6, alpha: float = ALPHA,
               n_restarts_optimizer: int = N_RESTARTS_OPTIMIZERS) -> CombinatorialStationaryBandit:
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param bandit_name: the name of the bandit for the experiment
    :param campaign: the campaign over which the bandit needs to optimize the budget allocation
    :param init_std_dev: GP parameter
    :param alpha: GP parameter
    :param n_restarts_optimizer: GP parameter
    :return: bandit that will be used to carry out the experiment
    """
    if bandit_name == "GPBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), init_std_dev, alpha, n_restarts_optimizer,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CombinatorialStationaryBandit(campaign=campaign, model_list=model_list)
    elif bandit_name == "GaussianBandit":
        model_list: List[DiscreteRegressor] = [DiscreteGaussianRegressor(list(campaign.get_budgets()),
                                                                         init_std_dev)
                                               for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CombinatorialStationaryBandit(campaign=campaign, model_list=model_list)
    elif bandit_name == "GPSWBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), init_std_dev, alpha, n_restarts_optimizer,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = SWCombinatorialBandit(campaign=campaign, model_list=model_list, sw_size=args.sw_size)
    elif bandit_name == "CDBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), init_std_dev, alpha, n_restarts_optimizer,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CDCombinatorialBandit(campaign=campaign, model_list=model_list, n_arms=N_ARMS,
                                       gamma=args.gamma, cd_threshold=args.cd_threshold, sw_size=args.sw_size)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def main(args):
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = AdvertisingEnvironment(scenario)

    campaign = Campaign(args.n_subcampaigns, args.cum_budget, args.n_arms)
    bandit = get_bandit(bandit_name=args.bandit_name, campaign=campaign)
    budget_allocation = [0, 0, 0]

    for t in range(0, args.n_rounds):
        # Choose arm
        budget_allocation_indexes = bandit.pull_arm()
        budget_allocation = [int(campaign.get_budgets()[i]) for i in budget_allocation_indexes]

        # Observe reward
        rewards = env.round(budget_allocation=budget_allocation)

        # Update bandit
        bandit.update(pulled_arm=budget_allocation_indexes, reward=rewards)

    return bandit.collected_rewards, budget_allocation


def run(id, seed, args):
    """
    Run a task to carry out the experiment

    :param id: id of the task
    :param seed: random seed that is used in this execution
    :param args: arguments given to the experiment
    :return: collected rewards
    """
    # Eventually fix here the seeds for additional sources of randomness (e.g. tensorflow)
    np.random.seed(seed)
    print("Starting run {}".format(id))
    rewards, best_allocation = main(args=args)
    print("Done run {}".format(id))
    return rewards, best_allocation


# Scheduling runs: ENTRY POINT
args = get_arguments()

seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
rewards = []
best_allocations = []
if args.n_jobs == 1:
    results = [run(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds)]
else:
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds))
for r in results:
    rewards.append(r[0])
    best_allocations.append(r[1])

c = Counter([tuple(x) for x in best_allocations])
best_allocation = list(c.most_common(1)[0][0])

if args.save_result:
    # Set up writing folder and file
    fd, folder_path_with_date = handle_folder_creation(result_path=args.output_folder)

    # Writing results and experiment details
    with open("{}reward_{}.pkl".format(folder_path_with_date, args.bandit_name), "wb") as output:
        pickle.dump(results, output)

    fd.write("Bandit experiment\n")
    fd.write("Number of arms: {}\n".format(args.n_arms))
    fd.write("Number of runs: {}\n".format(args.n_runs))
    fd.write("Horizon: {}\n".format(args.n_rounds))
    fd.write("Bandit algorithm: {}\n".format(args.bandit_name))
    fd.write("Scenario name {}\n".format(args.scenario_name))

    fd.write("Best Budget Allocation {}\n".format(str(best_allocation)))

    fd.close()

    # Plot cumulative regret and instantaneous reward
    rewards = np.mean(rewards, axis=0)
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = AdvertisingEnvironment(scenario)
    avg_regrets = []
    for reward in rewards:
        # The clairvoyance algorithm reward is the best reward he can get by sampling the environment
        # from the best budget allocation
        opt = sum(env.round(BEST_BUDGET_ALLOCATION))
        avg_regrets.append(opt - reward)
    cum_regrets = np.cumsum(avg_regrets)

    os.chdir(folder_path_with_date)

    plt.figure(0)
    plt.plot(cum_regrets, 'r')
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.suptitle("Budget Allocation - Combinatorial Bandit")
    plt.title(str(args.n_runs) + " Experiments - " + str(args.bandit_name))
    plt.savefig(fname="Regret.png", format="png")

    plt.figure(1)
    plt.plot(rewards, 'g')
    plt.xlabel("t")
    plt.ylabel("Instantaneous Reward")
    plt.suptitle("Budget Allocation - Combinatorial Bandit")
    plt.title(str(args.n_runs) + " Experiments - " + str(args.bandit_name))
    plt.savefig(fname="Reward.png", format="png")
