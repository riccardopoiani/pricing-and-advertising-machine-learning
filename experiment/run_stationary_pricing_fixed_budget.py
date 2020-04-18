import argparse
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from environments.GeneralEnvironment import PricingAdvertisingJointEnvironment
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.experiments_helper import build_discrete_bandit
from utils.folder_management import handle_folder_creation

# Basic default settings
N_ROUNDS = 100000
BASIC_OUTPUT_FOLDER = "../report/project_point_4/"

# Pricing settings
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"
MIN_PRICE = 15
MAX_PRICE = 25
N_ARMS = 11
DEFAULT_DISCRETIZATION = "UNIFORM"
FIXED_BUDGET = 1000 / 3
UNIT_COST = 12

CRP_UPPER_BOUND = 0.3
GAMMA_EXP_3 = 0.1


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()
    # Pricing settings
    parser.add_argument("-n_arms", "--n_arms", help="Number of arms for the prices", type=int, default=N_ARMS)
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
    parser.add_argument("-t", "--n_rounds", default=N_ROUNDS, help="Number of rounds", type=int)

    # Bandit hyper-parameters
    parser.add_argument("-b", "--bandit_name", help="Name of the bandit to be used in the experiment")
    parser.add_argument("-gamma", "--gamma",
                        help="Parameter for tuning the desire to pick an action uniformly at random",
                        type=float, default=GAMMA_EXP_3)
    parser.add_argument("-crp_ub", "--crp_upper_bound", help="Upper bound of the conversion rate probability",
                        type=float, default=CRP_UPPER_BOUND)

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=1)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)

    return parser.parse_args()


def get_prices(args):
    if args.discretization == "UNIFORM":
        return np.linspace(start=MIN_PRICE, stop=MAX_PRICE, num=args.n_arms)
    else:
        raise NotImplemented("Not implemented discretization method")


def main(args):
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = PricingAdvertisingJointEnvironment(scenario)
    env.set_budget_allocation([args.budget] * scenario.get_n_subcampaigns())

    prices = get_prices(args=args)
    arm_profit = prices - args.unit_cost
    bandit = build_discrete_bandit(bandit_name=args.bandit_name, n_arms=len(arm_profit),
                                   arm_values=arm_profit, args=args)

    iterate = not env.next_day()
    while iterate:
        iterate = not env.next_day()

    for _ in range(0, args.n_rounds):
        # Choose arm
        price_idx = bandit.pull_arm()

        # Observe reward
        env.next_user()
        reward, elapsed_day = env.round(price=prices[price_idx])
        reward = reward * arm_profit[price_idx]

        if elapsed_day:
            iterate = not env.next_day()
            while iterate:
                iterate = not env.next_day()

        # Update bandit
        bandit.update(pulled_arm=price_idx, observed_reward=reward)

    return bandit.collected_rewards, env.get_day_breakpoints()


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
    rewards, day_breakpoints = main(args=args)
    print("Done run {}".format(id))
    return rewards, day_breakpoints


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

    fd.write("Bandit experiment\n")
    fd.write("Number of arms: {}\n".format(args.n_arms))
    fd.write("Number of runs: {}\n".format(args.n_runs))
    fd.write("Horizon: {}\n".format(args.n_rounds))
    fd.write("Bandit algorithm: {}\n\n".format(args.bandit_name))
    fd.write("Scenario name {}\n".format(args.scenario_name))

    fd.write("Discretization type {}\n\n".format(args.discretization))

    fd.write("Bandit parameters \n")
    fd.write("Gamma parameter (EXP3) {}\n".format(args.gamma))
    fd.write("CRP Upper bound {}\n".format(args.crp_upper_bound))

    fd.close()
