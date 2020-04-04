import argparse
import os
import pickle
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from advertising.data_structure.Campaign import Campaign
from advertising.regressors.DiscreteGPRegressor import DiscreteGPRegressor
from advertising.regressors.DiscreteGaussianRegressor import DiscreteGaussianRegressor
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.combinatiorial.CDCombinatorialBandit import CDCombinatorialBandit
from bandit.combinatiorial.CombinatorialBandit import CombinatorialBandit
from bandit.combinatiorial.CombinatorialStationaryBandit import CombinatorialStationaryBandit
from bandit.combinatiorial.SWCombinatorialBandit import SWCombinatorialBandit
from bandit.joint.JointBandit import JointBandit
from environments.GeneralEnvironment import PricingAdvertisingJointEnvironment
from bandit.discrete import DiscreteBandit
from bandit.discrete.EXP3Bandit import EXP3Bandit
from bandit.discrete.TSBanditRescaledBernoulli import TSBanditRescaledBernoulli
from bandit.discrete.UCB1MBandit import UCB1MBandit
from bandit.discrete.UCBLBandit import UCBLBandit
from bandit.discrete.UCBLM import UCBLMBandit
from bandit.discrete.UCB1Bandit import UCB1Bandit
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation
from bandit.joint.IJointBandit import IJointBandit

# Basic default settings
N_DAYS = 100
BASIC_OUTPUT_FOLDER = "../report/project_point_4/"

# Scenario
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"

# Ads setting
N_ARMS_ADS = 11
CUM_BUDGET = 10000

# Ads bandit parameters
ALPHA = 100
N_RESTARTS_OPTIMIZERS = 10
INIT_STD: float = 1e6

# Pricing settings
MIN_PRICE = 15
MAX_PRICE = 25
N_ARMS_PRICE = 10
UNIT_COST = 12

# Pricing bandit parameters
GAMMA = 0.1
CRP_UPPER_BOUND = 0.1


def get_arguments():
    """
    Defining the arguments available for the script

    :return: argument parser
    """
    parser = argparse.ArgumentParser()

    # Joint bandit
    parser.add_argument("-jb", "--joint_bandit_name", help="Name of the bandit that jointly optimize"
                                                           "price and advertising", type=str)

    # Pricing settings
    parser.add_argument("-n_arms_price", "--n_arms_price", help="Number of arms for the prices", type=int,
                        default=N_ARMS_PRICE)
    parser.add_argument("-c", "--unit_cost", help="Unit cost for each sold unit", type=float, default=UNIT_COST)

    # Ads settings
    parser.add_argument("-n_arms_ads", "--n_arms_ads", help="Number of arms to be used for advertising", type=int,
                        default=N_ARMS_ADS)
    parser.add_argument("-bud", "--cum_budget", default=CUM_BUDGET,
                        help="Cumulative budget to be used for the simulation",
                        type=float)

    # Scenario settings
    parser.add_argument("-scenario_name", "--scenario_name", default=SCENARIO_NAME,
                        type=str, help="Name of the setting of the experiment")

    # Experiment run details
    parser.add_argument("-n_runs", "--n_runs", default=1, help="Number of runs of the experiments", type=int)
    parser.add_argument("-n_jobs", "--n_jobs", default=1, help="Number of jobs to be run", type=int)
    parser.add_argument("-t", "--n_days", default=N_DAYS, help="Number of rounds", type=int)

    # Ads bandit hyper-parameters
    parser.add_argument("-ab", "--ads_bandit_name", help="Name of the bandit to be used for advertising", type=str)
    parser.add_argument("-alpha", "--alpha", help="Alpha parameter for gaussian processes", type=int,
                        default=ALPHA)
    parser.add_argument("-n_restart_opt", "--n_restart_opt", help="Number of restarts for gaussian processes", type=int,
                        default=N_RESTARTS_OPTIMIZERS)
    parser.add_argument("-init_std", "--init_std", help="Initial standard deviation for gaussian processes",
                        type=float, default=INIT_STD)

    # Pricing bandit hyper-parameters
    parser.add_argument("-pb", "--pricing_bandit_name", help="Name of the bandit to be used for pricing", type=str)
    parser.add_argument("-gamma", "--gamma",
                        help="Parameter for tuning the desire to pick an action uniformly at random",
                        type=float, default=GAMMA)
    parser.add_argument("-crp_ub", "--crp_upper_bound", help="Upper bound of the conversion rate probability",
                        type=float, default=CRP_UPPER_BOUND)

    # Store results
    parser.add_argument("-s", "--save_result", help="Whether to store results or not", type=lambda x: int(x) != 0,
                        default=0)
    parser.add_argument("-o", "--output_folder", default=BASIC_OUTPUT_FOLDER, help="Basic folder where"
                                                                                   "to store the output",
                        type=str)

    return parser.parse_args()


def get_prices(args):
    return np.linspace(start=MIN_PRICE, stop=MAX_PRICE, num=args.n_arms_price)


def get_ads_bandit(args, campaign: Campaign) -> CombinatorialBandit:
    bandit_name = args.ads_bandit_name

    if bandit_name == "GPBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), args.init_std, args.alpha, args.n_restart_opt,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CombinatorialStationaryBandit(campaign=campaign, model_list=model_list)
    elif bandit_name == "GaussianBandit":
        model_list: List[DiscreteRegressor] = [DiscreteGaussianRegressor(list(campaign.get_budgets()),
                                                                         args.init_std)
                                               for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CombinatorialStationaryBandit(campaign=campaign, model_list=model_list)
    elif bandit_name == "GPSWBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), args.init_std, args.alpha, args.n_restart_opt,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = SWCombinatorialBandit(campaign=campaign, model_list=model_list, sw_size=args.sw_size)
    elif bandit_name == "CDBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), args.init_std, args.alpha, args.n_restart_opt,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CDCombinatorialBandit(campaign=campaign, model_list=model_list, n_arms=args.n_arms_ads,
                                       gamma=args.gamma, cd_threshold=args.cd_threshold, sw_size=args.sw_size)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def get_price_bandit(args, arm_values: np.array) -> DiscreteBandit:
    bandit_name = args.pricing_bandit_name

    if bandit_name == "TS":
        bandit = TSBanditRescaledBernoulli(n_arms=args.n_arms_price, arm_values=arm_values)
    elif bandit_name == "UCB1":
        bandit = UCB1Bandit(n_arms=args.n_arms_price, arm_values=arm_values)
    elif bandit_name == "UCB1M":
        bandit = UCB1MBandit(n_arms=args.n_arms_price, arm_values=arm_values)
    elif bandit_name == "UCBL":
        bandit = UCBLBandit(n_arms=args.n_arms_price, crp_upper_bound=args.crp_upper_bound, arm_values=arm_values)
    elif bandit_name == "UCBLM":
        bandit = UCBLMBandit(n_arms=args.n_arms_price, crp_upper_bound=args.crp_upper_bound, arm_values=arm_values)
    elif bandit_name == "EXP3":
        bandit = EXP3Bandit(n_arms=args.n_arms_price, gamma=args.gamma, arm_values=arm_values)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def get_bandit(args, arm_values: np.array, campaign: Campaign) -> IJointBandit:
    bandit_name = args.joint_bandit_name
    ads_bandit = get_ads_bandit(args=args, campaign=campaign)
    price_bandit_list = [get_price_bandit(args=args, arm_values=arm_values)
                         for _ in range(campaign.get_n_sub_campaigns())]

    if bandit_name == "JB":
        bandit = JointBandit(ads_learner=ads_bandit, price_learner=price_bandit_list, campaign=campaign)
    else:
        raise argparse.ArgumentParser("The name of the bandit to be used is not in the available ones")

    return bandit


def main(args):
    # Retrieve scenario
    scenario = EnvironmentManager.load_scenario(args.scenario_name)

    # Retrieve bandit and basic information
    campaign = Campaign(scenario.get_n_subcampaigns(), args.cum_budget, args.n_arms_ads)
    prices = get_prices(args=args)
    arm_profit = prices - args.unit_cost
    bandit = get_bandit(args=args, arm_values=arm_profit, campaign=campaign)

    # Create environment
    env = PricingAdvertisingJointEnvironment(scenario=scenario)

    for t in range(0, args.n_days):
        print("day {}".format(t))
        # Fix budget
        elapsed_day = False

        # Fix the price and simulate the day
        curr_budget_idx = bandit.pull_budget()
        curr_budget = [int(campaign.get_budgets()[i]) for i in curr_budget_idx]
        env.set_budget_allocation(budget_allocation=curr_budget)
        env.next_day()

        while not elapsed_day:
            # Retrieve user class
            user_class = env.next_user()[1]

            # Choose arm
            price_idx = bandit.pull_price(user_class=user_class)

            # Observe reward
            reward, elapsed_day = env.round(price=prices[price_idx])
            reward = reward * arm_profit[price_idx]

            # Update bandit
            bandit.update_price(pulled_arm=price_idx, user_class=user_class, observed_reward=reward)

        # The day is over, update the budgeting model
        bandit.update_budget(pulled_arm_list=curr_budget_idx, n_visits=env.get_daily_visits_per_sub_campaign())

    return bandit.get_daily_reward(), \
           bandit.get_reward_per_sub_campaign(), \
           bandit.get_daily_number_of_visit_per_sub_campaign(),\
           env.day_breakpoints


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
    total_rewards, price_rewards, ads_rewards, day_breakpoint = main(args=args)
    print("Done run {}".format(id))
    return total_rewards, price_rewards, ads_rewards, day_breakpoint


# Scheduling runs: ENTRY POINT
args = get_arguments()

seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
if args.n_jobs == 1:
    results = [run(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds)]
else:
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds))

total_rewards = [res[0] for res in results]
price_rewards = [res[1] for res in results]
ads_rewards = [res[2] for res in results]
day_breakpoint = [res[3] for res in results]

if args.save_result:
    # Set up writing folder and file
    fd, folder_path_with_date = handle_folder_creation(result_path=args.output_folder)

    # Writing results and experiment details
    print("Storing results on file...", end="")
    with open("{}total_reward_{}.pkl".format(folder_path_with_date, args.joint_bandit_name), "wb") as output:
        pickle.dump(total_rewards, output)
    with open("{}price_reward_{}.pkl".format(folder_path_with_date, args.pricing_bandit_name), "wb") as output:
        pickle.dump(price_rewards, output)
    with open("{}ads_reward_{}.pkl".format(folder_path_with_date, args.ads_bandit_name), "wb") as output:
        pickle.dump(ads_rewards, output)
    with open("{}day_{}.pkl".format(folder_path_with_date, args.joint_bandit_name), "wb") as output:
        pickle.dump(day_breakpoint, output)
    print("Done")

    fd.write("Bandit experiment\n")
    fd.write("Number of pricing arms: {}\n".format(args.n_arms_price))
    fd.write("Number of ads arm: {}\n".format(args.n_arms_ads))
    fd.write("Number of runs: {}\n".format(args.n_runs))
    fd.write("Number of days: {}\n".format(args.n_days))
    fd.write("Scenario name {}\n".format(args.scenario_name))
    fd.write("Unit cost: {}\n".format(args.unit_cost))
    fd.write("Joint bandit: {}\n".format(args.joint_bandit_name))
    fd.write("Bandit algorithm for pricing: {}\n".format(args.pricing_bandit_name))
    fd.write("Bandit algorithm for ads: {}\n".format(args.ads_bandit_name))
    fd.write("User prices: {}\n".format(get_prices(args=args)))
    fd.write("Cumulative budget: {}\n\n".format(args.cum_budget))

    fd.write("Bandit parameters \n")
    fd.write("Gamma parameter (EXP3) {}\n".format(args.gamma))
    fd.write("CRP upper bound {}\n".format(args.crp_upper_bound))
    fd.write("Alpha GP: {}\n".format(args.alpha))
    fd.write("Number of GP optimizer restarts: {}\n".format(args.n_restart_opt))
    fd.write("Initial standard deviation GP: {}\n".format(args.init_std))

    fd.close()

    # Plot instantaneous reward
    rewards = np.mean(total_rewards, axis=0)
    scenario = EnvironmentManager.load_scenario(args.scenario_name)
    env = PricingAdvertisingJointEnvironment(scenario)

    os.chdir(folder_path_with_date)

    plt.plot(rewards, 'g')
    plt.xlabel("t")
    plt.ylabel("Instantaneous Reward")
    plt.title(str(args.n_runs) + " Experiments")
    plt.savefig(fname="Reward.png", format="png")

