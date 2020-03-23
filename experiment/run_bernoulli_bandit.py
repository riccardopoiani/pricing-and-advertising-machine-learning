import argparse
import os
import pickle
import sys
from typing import List

import numpy as np
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from bandit.discrete import DiscreteBandit
from bandit.discrete.TSBandit import TSBandit
from utils.folder_management import handle_folder_creation
from environments.BernoulliDiscreteBanditEnvironment import BernoulliDiscreteBanditEnv
from utils.stats.BernoulliDistribution import BernoulliDistribution
from bandit.discrete.UCB1Bandit import UCB1Bandit
from bandit.discrete.UCB1MBandit import UCB1MBandit
from bandit.discrete.UCBLBandit import UCBLBandit
from bandit.discrete.EXP3Bandit import EXP3Bandit
from bandit.discrete.GIROBernoulliBandit import GIROBernoulliBandit
from bandit.discrete.LinPHE import LinPHE

N_ARMS = 10

ARMS_PROBABILITIES_PARAMETERS = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
PRICE_LIST = (np.array([11, 13, 20, 30, 40, 45, 55, 60, 65, 70]) / 70).tolist()

N_ROUNDS = 1000
BASIC_OUTPUT_FOLDER = "../report/bernoulli_bandit/"
BANDIT_NAMES = ['TS', 'UCB1', 'UCB1M', 'UCBL', 'EXP3' , 'GIRO', 'LINPHE']


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_runs", "--n_runs", default=1, help="Number of runs of the experiments", type=int)
    parser.add_argument("-n_jobs", "--n_jobs", default=1, help="Number of jobs to be run", type=int)
    parser.add_argument("-t", "--n_rounds", default=N_ROUNDS, help="Number of rounds", type=int)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)
    parser.add_argument("-b", "--bandit_name", help="Name of the bandit to be used in the experiment")
    parser.add_argument("-gamma", "--gamma", help="Parameter tuning the desire to pick an action uniformly at random",
                        type=float, default=0.1)
    parser.add_argument("-crp_ub", "--crp_upper_bound", help="Upper bound of the conversion rate probability",
                        type=float, default=0.1)

    parser.add_argument("-a", "--perturbation_hp", help="Parameter for perturbing the history", type=float, default=0.0)
    parser.add_argument("-l", "--regularization", help="Regularization parameter", type=float, default=0.0)
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=0)
    return parser.parse_args()


def get_bandit(args) -> DiscreteBandit:
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param args: command line arguments
    :return: bandit that will be used to carry out the experiment
    """
    bandit_name = args.bandit_name

    if bandit_name == "TS":
        bandit = TSBandit(n_arms=N_ARMS)
    elif bandit_name == "UCB1":
        bandit = UCB1Bandit(n_arms=N_ARMS)
    elif bandit_name == "UCB1M":
        bandit = UCB1MBandit(n_arms=N_ARMS, price_list=PRICE_LIST)
    elif bandit_name == "UCBL":
        bandit = UCBLBandit(n_arms=N_ARMS, crp_upper_bound=args.crp_upper_bound, price_list=PRICE_LIST)
    elif bandit_name == "EXP3":
        bandit = EXP3Bandit(n_arms=N_ARMS, gamma=args.gamma)
    elif bandit_name == "GIRO":
        bandit = GIROBernoulliBandit(n_arms=N_ARMS, a=args.perturbation_hp, prices=PRICE_LIST)
    elif bandit_name == "LINPHE":
        features = np.zeros(shape=(N_ARMS, 2))
        for i in range(N_ARMS):
            features[i, 0] = PRICE_LIST[i]*ARMS_PROBABILITIES_PARAMETERS[i]
            features[i, 1] = 1

        bandit = LinPHE(n_arms=N_ARMS, perturbation=args.perturbation_hp,
                        regularization=args.regularization,
                        features=features, features_dim=2,
                        prices=np.array(PRICE_LIST))
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def main(args):
    # Environment creation
    probabilities: List[BernoulliDistribution] = []
    for p in ARMS_PROBABILITIES_PARAMETERS:
        probabilities.append(BernoulliDistribution(theta=p))

    env = BernoulliDiscreteBanditEnv(n_arms=N_ARMS, probabilities=probabilities)

    bandit = get_bandit(args=args)

    for t in range(0, args.n_rounds):
        # Choose arm
        idx = bandit.pull_arm()

        # Observe reward
        reward = env.round(pulled_arm=idx) * PRICE_LIST[idx]

        # Update bandit
        bandit.update(pulled_arm=idx, reward=reward)

    return bandit.collected_rewards


def run(id, seed, args):
    """
    Run a task to carry out the experiment

    :param id: id of the task
    :param seed: random seed that is used in this execution
    :param args: arguments given to the experiment
    :return: collected rewards
    """
    # Eventually fix here the seeds for additional sources of randomness (e.g. scipy)
    np.random.seed(seed)
    print("Starting run {}".format(id))
    rewards = main(args=args)
    print("Done run {}".format(id))
    return rewards


# Scheduling runs
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
    fd.write("Number of arms: {}\n".format(N_ARMS))
    fd.write("Probability distributions: {}\n".format(ARMS_PROBABILITIES_PARAMETERS))
    fd.write("Number of runs: {}\n".format(args.n_runs))
    fd.write("Horizon: {}\n".format(args.n_rounds))
    fd.write("Bandit algorithm: {}\n".format(args.bandit_name))
    fd.close()
