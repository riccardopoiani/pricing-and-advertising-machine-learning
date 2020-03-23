import argparse
import os
import pickle
import sys
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from advertising.data_structure.Campaign import Campaign
from bandit.discrete.CombinatorialBandit import CombinatorialBandit
from bandit.discrete.CombinatorialGPBandit import CombinatorialGPBandit
from bandit.discrete.CombinatorialGaussianBandit import CombinatorialGaussianBandit
from environments.AdEnvironment import AdEnvironment
from environments.Settings.AdvertisingScenario import PolynomialAdvertisingScenario

from environments.Settings import EnvironmentSettings
from utils.folder_management import handle_folder_creation

# Basic default settings
N_ROUNDS = 100
BASIC_OUTPUT_FOLDER = "../report/project_point_2/"
EXPERIMENT_DEFAULT_SETTING = "POLYNOMIAL_ADVERTISING_SCENARIO"

# Advertising settings
CUM_BUDGET = 100
MIN_BUDGET = 0
MAX_BUDGET = CUM_BUDGET
N_SUBCAMPAIGNS = 3
N_ARMS = 21

MAX_N_CLICKS = [3000, 2000, 1500]
LINEAR_COEFFICIENTS = [(30 + i * 10) for i in range(N_SUBCAMPAIGNS)]
EXPONENTS = [1] * N_SUBCAMPAIGNS
# BIASES = [1000 * (N_SUBCAMPAIGNS + 1 - i) for i in range(1, N_SUBCAMPAIGNS + 1)]
BIASES = [0] * N_SUBCAMPAIGNS
SIGMAS = [1] * N_SUBCAMPAIGNS
BEST_BUDGETS = [20.0, 50.0, 30.0]

# 3*x1 + 0 = 300 max
# 4*x2 + 0 = 400 max
# 5*x3 + 0 = 500 max


POLYNOMIAL_ADVERTISING_SCENARIO_KWARGS = {
    "linear_coefficients": LINEAR_COEFFICIENTS,
    "exponents": EXPONENTS,
    "biases": BIASES,
    "max_n_clicks_list": MAX_N_CLICKS
}


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
    parser.add_argument("-min_bud", "--min_budget", default=MIN_BUDGET,
                        help="Minimum budget to be used for the simulation",
                        type=float)
    parser.add_argument("-max_bud", "--max_budget", default=MAX_BUDGET,
                        help="Maximum budget to be used for the simulation",
                        type=float)
    parser.add_argument("-n_sub", "--n_subcampaigns", default=N_SUBCAMPAIGNS,
                        help="Number of subcampaigns to be used for the simulation",
                        type=float)
    parser.add_argument("-n_arms", "--n_arms", help="Number of arms for the prices", type=int, default=N_ARMS)

    # Scenario settings
    parser.add_argument("-scenario_name", "--scenario_name", default=EXPERIMENT_DEFAULT_SETTING,
                        type=str, help="Name of the setting of the experiment")

    # Experiment run details
    parser.add_argument("-n_runs", "--n_runs", default=1, help="Number of runs of the experiments", type=int)
    parser.add_argument("-n_jobs", "--n_jobs", default=1, help="Number of jobs to be run", type=int)
    parser.add_argument("-t", "--n_rounds", default=N_ROUNDS, help="Number of rounds", type=int)

    # Bandit hyper-parameters
    parser.add_argument("-b", "--bandit_name", help="Name of the bandit to be used in the experiment")

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=0)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)

    return parser.parse_args()


def get_scenario_kwargs(scenario_name: str):
    if scenario_name == PolynomialAdvertisingScenario.get_scenario_name():
        return POLYNOMIAL_ADVERTISING_SCENARIO_KWARGS


def get_bandit(bandit_name: str, campaign: Campaign, init_std_dev: float = 1e3, alpha: float = 10,
               n_restarts_optimizer: int = 5) -> CombinatorialBandit:
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
        bandit = CombinatorialGPBandit(campaign=campaign, init_std_dev=init_std_dev, alpha=alpha,
                                       n_restarts_optimizer=n_restarts_optimizer)
    elif bandit_name == "GaussianBandit":
        bandit = CombinatorialGaussianBandit(campaign=campaign, init_std_dev=init_std_dev)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def main(args):
    number_of_clicks_per_budget_list = EnvironmentSettings.EnvironmentManager.get_setting(args.scenario_name,
                                                                                          **get_scenario_kwargs(
                                                                                              args.scenario_name))

    env = AdEnvironment(list(number_of_clicks_per_budget_list), SIGMAS)

    campaign = Campaign(args.n_subcampaigns, args.cum_budget, args.n_arms)
    bandit = get_bandit(bandit_name=args.bandit_name, campaign=campaign)

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
    fd.write("Bandit algorithm: {}\n\n".format(args.bandit_name))
    fd.write("Scenario name {}\n".format(args.scenario_name))

    fd.write("Polynomial Advertising Scenario {}\n\n".format(POLYNOMIAL_ADVERTISING_SCENARIO_KWARGS))

    fd.write("Best Budget Allocation {}\n".format(str(best_allocation)))

    fd.close()

    rewards = np.mean(rewards, axis=0)
    number_of_clicks_per_budget_list = EnvironmentSettings.EnvironmentManager.get_setting(args.scenario_name,
                                                                                          **get_scenario_kwargs(
                                                                                              args.scenario_name))
    avg_regrets = []
    for reward in rewards:
        opt = max(reward, sum(number_of_clicks_per_budget_list[i](BEST_BUDGETS[i])
                              for i in range(int(args.n_subcampaigns))))
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