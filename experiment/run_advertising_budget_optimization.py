import argparse
import os
import pickle
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from utils.experiments_helper import build_combinatorial_bandit
from environments.GeneralEnvironment import PricingAdvertisingJointEnvironment
from advertising.data_structure.Campaign import Campaign
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation

# Basic default settings
N_ROUNDS = 100
BASIC_OUTPUT_FOLDER = "../report/project_point_2/"

# Bandit parameters
ALPHA = 100
N_RESTARTS_OPTIMIZERS = 10
INIT_STD = 1e6

# Advertising settings
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"
CUM_BUDGET = 10000
N_ARMS = 11


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
    parser.add_argument("-alpha", "--alpha", help="Alpha parameter for gaussian processes", type=int,
                        default=ALPHA)
    parser.add_argument("-n_restart_opt", "--n_restart_opt", help="Number of restarts for gaussian processes", type=int,
                        default=N_RESTARTS_OPTIMIZERS)
    parser.add_argument("-init_std", "--init_std", help="Initial standard deviation for regressors",
                        type=float, default=INIT_STD)
    parser.add_argument("-sw", "--sw_size", help="Size of the sliding window for SW bandits", type=int)
    parser.add_argument("-gamma", "--gamma",
                        help="Controls the fraction of the uniform sampling over the number of arms",
                        type=float, default=0.1)
    parser.add_argument("-cd_threshold", "--cd_threshold", help="Threshold used for change detection",
                        type=float, default=0.1)

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=1)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)

    return parser.parse_args()


def main(args):
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = PricingAdvertisingJointEnvironment(scenario)

    campaign = Campaign(scenario.get_n_subcampaigns(), args.cum_budget, args.n_arms)
    bandit = build_combinatorial_bandit(bandit_name=args.bandit_name, campaign=campaign,
                                        init_std=args.init_std, args=args)
    budget_allocation = [0, 0, 0]

    for t in range(0, args.n_rounds):
        # Choose arm
        budget_allocation_indexes = bandit.pull_arm()
        budget_allocation = [int(campaign.get_budgets()[i]) for i in budget_allocation_indexes]

        # Observe reward
        env.set_budget_allocation(budget_allocation=budget_allocation)
        env.next_day()
        rewards = env.get_daily_visits_per_sub_campaign()

        # Update bandit
        bandit.update(pulled_arm=budget_allocation_indexes, bernoulli_sample=rewards)

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

    # Plot instantaneous reward
    rewards = np.mean(rewards, axis=0)
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = PricingAdvertisingJointEnvironment(scenario)

    os.chdir(folder_path_with_date)

    plt.plot(rewards, 'g')
    plt.xlabel("t")
    plt.ylabel("Instantaneous Reward")
    plt.suptitle("Budget Allocation - Combinatorial Bandit")
    plt.title(str(args.n_runs) + " Experiments - " + str(args.bandit_name))
    plt.savefig(fname="Reward.png", format="png")
