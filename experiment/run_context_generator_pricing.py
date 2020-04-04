import argparse
import os
import pickle
import sys
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from bandit.context.ContextualBandit import ContextualBandit
from environments.PricingEnvironmentContextGeneration import PricingEnvironmentContextGeneration
from bandit.discrete import DiscreteBandit
from bandit.discrete.EXP3Bandit import EXP3Bandit
from bandit.discrete.TSBanditRescaledBernoulli import TSBanditRescaledBernoulli
from bandit.discrete.UCB1MBandit import UCB1MBandit
from bandit.discrete.UCBLBandit import UCBLBandit
from bandit.discrete.UCBLM import UCBLMBandit
from bandit.discrete.UCB1Bandit import UCB1Bandit
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation

# Basic default settings
N_DAYS = 7 * 10
CONFIDENCE = 0.99
CONTEXT_GENERATION_FREQUENCY = 7
BASIC_OUTPUT_FOLDER = "../report/project_point_5/"

# Pricing settings
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"
MIN_PRICE = 0
MAX_PRICE = 25
N_ARMS = 10
DEFAULT_DISCRETIZATION = "UNIFORM"
FIXED_BUDGET = 1000 / 3
UNIT_COST = 12


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()
    # Pricing settings
    parser.add_argument("-n_arms", "--n_arms", help="Number of arms for the prices", type=int, default=N_ARMS)
    parser.add_argument("-conf", "--confidence", help="Confidence for Hoeffding bound", type=float, default=CONFIDENCE)
    parser.add_argument("-freq", "--context_gen_freq", help="Context generation frequency", type=int,
                        default=CONTEXT_GENERATION_FREQUENCY)
    parser.add_argument("-d", "--discretization", help="Discretization type", type=str, default=DEFAULT_DISCRETIZATION)
    parser.add_argument("-c", "--unit_cost", help="Unit cost for each sold unit", type=float, default=UNIT_COST)

    # Ads setting
    parser.add_argument("-bud", "--budget", default=FIXED_BUDGET, help="Fixed budget to be used for the simulation",
                        type=float)

    # Scenario settings
    parser.add_argument("-scenario_name", "--scenario_name", default=SCENARIO_NAME,
                        type=str, help="Name of the setting of the experiment")

    # Experiment run details
    parser.add_argument("-n_runs", "--n_runs", default=1, help="Number of runs of the experiments", type=int)
    parser.add_argument("-n_jobs", "--n_jobs", default=1, help="Number of jobs to be run", type=int)
    parser.add_argument("-t", "--n_days", default=N_DAYS, help="Number of days", type=int)

    # Bandit hyper-parameters
    parser.add_argument("-b", "--bandit_name", help="Name of the bandit to be used in the experiment")
    parser.add_argument("-gamma", "--gamma",
                        help="Parameter for tuning the desire to pick an action uniformly at random",
                        type=float, default=0.1)
    parser.add_argument("-crp_ub", "--crp_upper_bound", help="Upper bound of the conversion rate probability",
                        type=float, default=0.2)
    parser.add_argument("-a", "--perturbation", help="Parameter for perturbing the history", type=float, default=0.0)
    parser.add_argument("-l", "--regularization", help="Regularization parameter", type=float, default=0.0)

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=0)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)

    return parser.parse_args()


def get_prices(args):
    if args.discretization == "UNIFORM":
        return np.linspace(start=MIN_PRICE, stop=MAX_PRICE, num=args.n_arms)
    else:
        raise NotImplemented("Not implemented discretization method")


def get_bandit(args, arm_values: np.array, N_FEATURES: int = 2) -> DiscreteBandit:
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param args: command line arguments
    :param arm_values: values of each arm
    :return: bandit that will be used to carry out the experiment
    """
    bandit_name = args.bandit_name

    if bandit_name == "TS":
        bandit_class = TSBanditRescaledBernoulli
        bandit_kwargs = {"n_arms": N_ARMS, "arm_values": arm_values}
    elif bandit_name == "UCB1":
        bandit_class = UCB1Bandit
        bandit_kwargs = {"n_arms": N_ARMS, "arm_values": arm_values}
    elif bandit_name == "UCB1M":
        bandit_class = UCB1MBandit
        bandit_kwargs = {"n_arms": N_ARMS, "arm_values": arm_values}
    elif bandit_name == "UCBL":
        bandit_class = UCBLBandit
        bandit_kwargs = {"n_arms": N_ARMS, "crp_upper_bound": args.crp_upper_bound, "arm_values": arm_values}
    elif bandit_name == "UCBLM":
        bandit_class = UCBLMBandit
        bandit_kwargs = {"n_arms": N_ARMS, "crp_upper_bound": args.crp_upper_bound, "arm_values": arm_values}
    elif bandit_name == "EXP3":
        bandit_class = EXP3Bandit
        bandit_kwargs = {"n_arms": N_ARMS, "gamma": args.gamma, "arm_values": arm_values}
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    bandit = ContextualBandit(N_FEATURES, CONFIDENCE, CONTEXT_GENERATION_FREQUENCY, bandit_class, **bandit_kwargs)
    return bandit


def main(args):
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = PricingEnvironmentContextGeneration(scenario,
                                              fixed_budget_allocation=[args.budget] * scenario.get_n_subcampaigns())

    prices = get_prices(args=args)
    arm_profit = prices - args.unit_cost
    bandit = get_bandit(args=args, arm_values=prices)

    for i in range(0, args.n_days):
        # User arrival
        n_users = env.get_n_users()
        for j in range(n_users):
            # A user is specified by a min context (that is when all the features are specified)
            u: Tuple[int, ...] = env.get_user()
            # Choose arm
            price_idx = bandit.pull_arm(u)
            # Observe reward
            reward = env.round(price=prices[price_idx], min_context=u) * prices[price_idx]
            # Update bandit
            bandit.update(min_context=u, pulled_arm=price_idx, reward=reward)
        bandit.next_day()
        if i % 7 == 0:
            print("SPLIT WEEK {}: {}".format(i, bandit.context_structure))

    return bandit.collected_rewards, env.get_day_breakpoints(), bandit.context_structure


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
    rewards, day_breakpoints, context_structures = main(args=args)
    print("Done run {}".format(id))
    return rewards, day_breakpoints, context_structures


# Scheduling runs: ENTRY POINT
args = get_arguments()

seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
if args.n_jobs == 1:
    results = [run(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds)]
else:
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds))

rewards = [res[0] for res in results]
day_breakpoint = [res[1] for res in results]
context_structures = [res[2] for res in results]

if args.save_result:
    # Set up writing folder and file
    fd, folder_path_with_date = handle_folder_creation(result_path=args.output_folder)

    # Writing results and experiment details
    print("Storing results on file...", end="")
    with open("{}reward_{}.pkl".format(folder_path_with_date, args.bandit_name), "wb") as output:
        pickle.dump(rewards, output)
    with open("{}day_{}.pkl".format(folder_path_with_date, args.bandit_name), "wb") as output:
        pickle.dump(day_breakpoint, output)
    print("Done")

    fd.write("Context Generation Bandit experiment\n")
    fd.write("Number of arms: {}\n".format(args.n_arms))
    fd.write("Number of runs: {}\n".format(args.n_runs))
    fd.write("Horizon in days: {}\n".format(args.n_days))
    fd.write("Bandit algorithm: {}\n\n".format(args.bandit_name))
    fd.write("Scenario name {}\n".format(args.scenario_name))

    fd.write("Discretization type {}\n\n".format(args.discretization))

    fd.write("Bandit parameters \n")
    fd.write("Perturbation parameter (LinPHE; GIRO) {}\n".format(args.perturbation))
    fd.write("Regularization parameter (LinPHE) {}\n".format(args.regularization))
    fd.write("Gamma parameter (EXP3) {}\n".format(args.gamma))

    max_occurrence = 0
    context_structure_mode = ""
    context_structures_dict = defaultdict(int)
    for context_structure in context_structures:
        context_structures_dict[str(context_structure)] += 1
        if context_structures_dict[str(context_structure)] > max_occurrence:
            max_occurrence = context_structures_dict[str(context_structure)]
            context_structure_mode = context_structure

    fd.write("Context Generated: {}\n".format(context_structure_mode))

    fd.close()

    # Plot cumulative regret and instantaneous reward
    daily_rewards = np.zeros(shape=(args.n_runs, args.n_days), dtype=np.float)
    for e in range(args.n_runs):
        n_users = 0
        for d, d_b in enumerate(day_breakpoint[e]):
            daily_rewards[e, d] = sum(rewards[e][n_users:n_users + d_b])
            n_users += d_b

    rewards = np.mean(daily_rewards, axis=0)
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = PricingEnvironmentContextGeneration(scenario,
                                              fixed_budget_allocation=[args.budget] * scenario.get_n_subcampaigns())
    clicks_per_subcampaign = scenario.get_phases()[0].get_all_n_clicks(
        budget_allocation=[args.budget] * scenario.get_n_subcampaigns())
    total_revenue_per_price = np.zeros(shape=(args.n_arms, scenario.get_n_subcampaigns()))
    for i, price in enumerate(get_prices(args)):
        crps = EnvironmentManager.get_crps_for_prices(args.scenario_name, [price] * scenario.get_n_subcampaigns())[0]
        revenue = np.array(crps) * (price)
        total_revenue = np.array(clicks_per_subcampaign) * revenue
        total_revenue_per_price[i, :] = total_revenue

    opt_reward = sum(np.max(total_revenue_per_price, axis=0))

    avg_regrets = []
    for reward in rewards:
        # The clairvoyant algorithm reward is the best reward he can get by sampling the environment
        # from the best context allocation
        avg_regrets.append(opt_reward - reward)
    cum_regrets = np.cumsum(avg_regrets)

    os.chdir(folder_path_with_date)

    plt.figure(0)
    plt.plot(cum_regrets, 'r')
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.suptitle("Context Generation - REGRET")
    plt.title(str(args.n_runs) + " Experiments - " + str(args.bandit_name))
    plt.savefig(fname="Regret.png", format="png")

    plt.figure(1)
    plt.plot(rewards, 'g')
    plt.xlabel("t")
    plt.ylabel("Instantaneous Reward")
    plt.suptitle("Context Generation - REWARD")
    plt.title(str(args.n_runs) + " Experiments - " + str(args.bandit_name))
    plt.savefig(fname="Reward.png", format="png")
