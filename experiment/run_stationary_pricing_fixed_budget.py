import argparse
import os
import pickle
import sys

import numpy as np
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from environments.PricingStationaryEnvironmentFixedBudget import PricingStationaryEnvironmentFixedBudget
from bandit.discrete import DiscreteBandit
from bandit.discrete.EXP3Bandit import EXP3Bandit
from bandit.discrete.GIROBernoulliBandit import GIROBernoulliBandit
from bandit.discrete.LinPHE import LinPHE
from bandit.discrete.TSBanditRescaledBernoulli import TSBanditRescaledBernoulli
from bandit.discrete.UCB1Bandit import UCB1Bandit
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation

# Basic default settings
N_ROUNDS = 1000
BASIC_OUTPUT_FOLDER = "../report/project_point_4/"

# Pricing settings
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"
MIN_PRICE = 3
MAX_PRICE = 10
N_ARMS = 10
DEFAULT_DISCRETIZATION = "UNIFORM"
FIXED_BUDGET = 50


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()
    # Pricing settings
    parser.add_argument("-n_arms", "--n_arms", help="Number of arms for the prices", type=int, default=N_ARMS)
    parser.add_argument("-d", "--discretization", help="Discretization type", type=str, default=DEFAULT_DISCRETIZATION)

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
                        type=float, default=0.0)
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


def get_bandit(args, prices) -> DiscreteBandit:
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param args: command line arguments
    :return: bandit that will be used to carry out the experiment
    """
    bandit_name = args.bandit_name

    if bandit_name == "TS":
        bandit = TSBanditRescaledBernoulli(n_arms=args.n_arms, prices=prices)
    elif bandit_name == "UCB1":
        bandit = UCB1Bandit(n_arms=args.n_arms)
    elif bandit_name == "EXP3":
        bandit = EXP3Bandit(n_arms=args.n_arms, gamma=args.gamma)
    elif bandit_name == "GIRO":
        bandit = GIROBernoulliBandit(n_arms=args.n_arms, a=args.perturbation, prices=prices)
    elif bandit_name == "LINPHE":
        features = np.zeros(shape=(args.n_arms, 2))
        for i in range(args.n_arms):
            features[i, 0] = prices[i]
            features[i, 1] = 1

        bandit = LinPHE(n_arms=args.n_arms, perturbation=args.perturbation,
                        regularization=args.regularization,
                        features=features, features_dim=2, prices=prices)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def main(args):
    phases = EnvironmentManager.load_scenario(args.scenario_name)
    n_subcampaigns = phases[0].get_n_subcampaigns()
    env = PricingStationaryEnvironmentFixedBudget(n_subcampaigns, phases,
                                                  fixed_budget_allocation=[args.budget] * n_subcampaigns)

    prices = get_prices(args=args)
    bandit = get_bandit(args=args, prices=prices)

    for t in range(0, args.n_rounds):
        # Choose arm
        price_idx = bandit.pull_arm()

        # Observe reward
        reward = env.round(price=prices[price_idx]) * prices[price_idx]

        # Update bandit
        bandit.update(pulled_arm=price_idx, reward=reward)

    return bandit.collected_rewards


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
    rewards = main(args=args)
    print("Done run {}".format(id))
    return rewards


# Scheduling runs: ENTRY POINT
args = get_arguments()

seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
if args.n_jobs == 1:
    results = [run(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds)]
else:
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds))

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

    fd.write("Discretization type {}\n\n".format(args.discretization))

    fd.write("Bandit parameters \n")
    fd.write("Perturbation parameter (LinPHE; GIRO) {}\n".format(args.perturbation))
    fd.write("Regularization parameter (LinPHE) {}\n".format(args.regularization))
    fd.write("Gamma parameter (EXP3) {}\n".format(args.gamma))

    fd.close()
