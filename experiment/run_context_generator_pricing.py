import argparse
import os
import pickle
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from bandit.context.BruteforceContextGenerator import BruteforceContextGenerator
from bandit.context.ContextualBandit import ContextualBandit
from bandit.context.GreedyContextGenerator import GreedyContextGenerator

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from utils.experiments_helper import get_bandit_class_and_kwargs
from environments.GeneralEnvironment import PricingAdvertisingJointEnvironment
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation

# Basic default settings
N_DAYS = 100
CONFIDENCE = 0.01
CONTEXT_GENERATION_FREQUENCY = 7
CONTEXT_GENERATION_NAME = "GCG"
BASIC_OUTPUT_FOLDER = "../report/project_point_5/"

# Pricing settings
SCENARIO_NAME = "linear_visit_tanh_price"  # corresponds to the name of the file in "resources"
MIN_PRICE = 15
MAX_PRICE = 25
N_ARMS = 11
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
    parser.add_argument("-cgn", "--context_gen_name", help="Context generation name", type=str,
                        default=CONTEXT_GENERATION_NAME)
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


def get_bandit(bandit_name, n_arms, arm_values, n_features, context_generation_frequency,
               context_generator_name, args):
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param bandit_name: name of the bandit to use
    :param n_arms: number of arms of the bandit
    :param arm_values: values of each arm
    :param n_features: number of user features
    :param context_generation_frequency: the frequency in days of the context generation
    :param context_generator_name: name of the context generator to use
    :param args: command line arguments
    :return: bandit that will be used to carry out the experiment
    """
    bandit_class, bandit_kwargs = get_bandit_class_and_kwargs(bandit_name, n_arms, arm_values, args)
    if context_generator_name == "GCG":
        context_generator_class = GreedyContextGenerator
    elif context_generator_name == "BCG":
        context_generator_class = BruteforceContextGenerator
    else:
        raise argparse.ArgumentError("The name of the context generator to be used is not in the available ones")

    bandit = ContextualBandit(n_features, args.confidence, context_generation_frequency, context_generator_class,
                              bandit_class, **bandit_kwargs)
    return bandit


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
    bandit = get_bandit(bandit_name=args.bandit_name, n_arms=len(arm_profit),
                        arm_values=arm_profit, n_features=scenario.get_n_user_features(),
                        context_generation_frequency=args.context_gen_freq,
                        context_generator_name=args.context_gen_name, args=args)

    env.next_day()

    for i in range(0, args.n_days):
        # User arrival
        daily_n_users = int(np.sum(env.get_daily_visits_per_sub_campaign()))
        for j in range(daily_n_users):
            # A user is specified by a min context (that is when all the features are specified)
            user_min_context, _ = env.next_user()
            # Choose arm
            price_idx = bandit.pull_arm(user_min_context)
            # Observe reward
            reward, _ = env.round(price=prices[price_idx])
            reward = reward * arm_profit[price_idx]
            # Update bandit
            bandit.update(min_context=user_min_context, pulled_arm=price_idx, reward=reward)
        bandit.next_day()
        #if i % 7 == 0:
        #    print(bandit.context_structure)
        env.next_day()
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
    fd.write("Confidence: {}\n".format(args.confidence))

    fd.write("Bandit algorithm: {}\n".format(args.bandit_name))
    fd.write("Context generation algorithm: {}\n\n".format(args.context_gen_name))
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
    for exp in range(args.n_runs):
        for day in range(0, len(day_breakpoint[exp]) - 1):
            daily_rewards[exp, day] = np.sum(rewards[exp][day_breakpoint[exp][day]: day_breakpoint[exp][day + 1]])

    # Calculates optimal rewards (for clairvoyant algorithm)
    rewards = np.mean(daily_rewards, axis=0)
    os.chdir(folder_path_with_date)

    plt.figure(1)
    plt.plot(rewards, 'g')
    plt.xlabel("t")
    plt.ylabel("Instantaneous Reward")
    plt.suptitle("Context Generation - REWARD")
    plt.title(str(args.n_runs) + " Experiments - " + str(args.bandit_name))
    plt.savefig(fname="Reward.png", format="png")
