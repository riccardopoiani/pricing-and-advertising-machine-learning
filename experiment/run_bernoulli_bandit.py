import argparse
import pickle
import numpy as np
import os
import sys

from joblib import Parallel, delayed
from typing import List

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from bandit.discrete import DiscreteBandit
from utils.folder_management import handle_folder_creation
from environments.BernoulliDiscreteBanditEnv import BernoulliDiscreteBanditEnv
from utils.stats.BernoulliDistribution import BernoulliDistribution
from bandit.discrete.TSBanditBernoulli import TSBanditBernoulli
from bandit.discrete.UCB1Bandit import UCB1Bandit

N_ARMS = 2
ARMS_PROBABILITIES_PARAMETERS = [0.8, 0.1]
N_ROUNDS = 1000
BASIC_OUTPUT_FOLDER = "../report/bernoulli_bandit/"
BANDIT_NAMES = ['TS']


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
    return parser.parse_args()


def get_bandit(bandit_name: str) -> DiscreteBandit:
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param bandit_name: name of the bandit to be used
    :return: bandit that will be used to carry out the experiment
    """
    if bandit_name == "TS":
        bandit = TSBanditBernoulli(n_arms=N_ARMS)
    elif bandit_name == "UCB1":
        bandit = UCB1Bandit(n_arms=N_ARMS)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def main(args):
    # Environment creation
    probabilities: List[BernoulliDistribution] = []
    for p in ARMS_PROBABILITIES_PARAMETERS:
        probabilities.append(BernoulliDistribution(theta=p))

    env = BernoulliDiscreteBanditEnv(n_arms=N_ARMS, probabilities=probabilities)

    bandit = get_bandit(args.bandit_name)

    for t in range(0, args.n_rounds):
        # Choose arm
        idx_ts = bandit.pull_arm()

        # Observe reward
        reward_ts = env.round(pulled_arm=idx_ts)

        # Update bandit
        bandit.update(pulled_arm=idx_ts, reward=reward_ts)

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
