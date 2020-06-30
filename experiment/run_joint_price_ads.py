import argparse
import os
import pickle
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.append("../")

from bandit.joint.JointBanditDiscriminatoryImproved import JointBanditDiscriminatoryImproved
from bandit.joint.AdValueStrategy import QuantileAdValueStrategy, ExpectationAdValueStrategy
from bandit.joint.JointBanditFixedDailyPriceQuantile import JointBanditFixedDailyPriceQuantile
from bandit.joint.JointBanditFixedDailyPriceTS import JointBanditFixedDailyPriceTS
from bandit.joint.JointBanditDiscriminatory import JointBanditDiscriminatory
from bandit.joint.JointBanditBalanced import JointBanditBalanced
from advertising.data_structure.Campaign import Campaign
from advertising.regressors.DiscreteGPRegressor import DiscreteGPRegressor
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from environments.GeneralEnvironment import PricingAdvertisingJointEnvironment
from environments.Settings.EnvironmentManager import EnvironmentManager
from utils.folder_management import handle_folder_creation
from utils.experiments_helper import build_combinatorial_bandit, get_bandit_class_and_kwargs
from bandit.joint.IJointBandit import IJointBandit

# Basic default settings
N_DAYS = 50
BASIC_OUTPUT_FOLDER = "../report/project_point_7/change_std/500"

# Scenario
SCENARIO_NAME = "linear_scenario"  # corresponds to the name of the file in "resources"

# Joint parameters
MIN_STD_QUANTILE = 0.1

# Ads setting
N_ARMS_ADS = 11
CUM_BUDGET = 1000
DEFAULT_ADS_BANDIT = "GPBandit"

# Ads bandit parameters
ALPHA = 100
N_RESTARTS_OPTIMIZERS = 10
INIT_STD: float = 1e6

# Pricing settings
DEFAULT_PRICE_BANDIT = "TS"
MIN_PRICE = 15
MAX_PRICE = 25
N_ARMS_PRICE = 11
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
    parser.add_argument("-daily_price", "--daily_price", default=0, help="Pull a daily price or a price for each user",
                        type=int)

    # Joint bandit hyper-parameters
    parser.add_argument("-min_std_q", "--min_std_q", default=MIN_STD_QUANTILE, type=float, help="Minimum standard"
                                                                                                "deviation"
                                                                                                "for quantile"
                                                                                                "calculation")

    # Ads bandit hyper-parameters
    parser.add_argument("-ab", "--ads_bandit_name", help="Name of the bandit to be used for advertising", type=str,
                        default=DEFAULT_ADS_BANDIT)
    parser.add_argument("-alpha", "--alpha", help="Alpha parameter for gaussian processes", type=int,
                        default=ALPHA)
    parser.add_argument("-n_restart_opt", "--n_restart_opt", help="Number of restarts for gaussian processes", type=int,
                        default=N_RESTARTS_OPTIMIZERS)
    parser.add_argument("-init_std", "--init_std", help="Initial standard deviation for regressors",
                        type=float, default=INIT_STD)

    # Pricing bandit hyper-parameters
    parser.add_argument("-pb", "--pricing_bandit_name", help="Name of the bandit to be used for pricing", type=str,
                        default=DEFAULT_PRICE_BANDIT)
    parser.add_argument("-gamma", "--gamma",
                        help="Parameter for tuning the desire to pick an action uniformly at random",
                        type=float, default=GAMMA)
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
    return np.linspace(start=MIN_PRICE, stop=MAX_PRICE, num=args.n_arms_price)


def get_bandit(args, arm_values: np.array, campaign: Campaign) -> IJointBandit:
    bandit_name: str = args.joint_bandit_name
    ads_bandit = build_combinatorial_bandit(bandit_name=args.ads_bandit_name, campaign=campaign,
                                            init_std=args.init_std, args=args)
    price_bandit_class, price_bandit_kwargs = get_bandit_class_and_kwargs(bandit_name=args.pricing_bandit_name,
                                                                          n_arms=len(arm_values),
                                                                          arm_values=arm_values, args=args)
    price_bandit_list = [price_bandit_class(**price_bandit_kwargs)
                         for _ in range(campaign.get_n_sub_campaigns())]

    ad_value_strategy = ExpectationAdValueStrategy(np.max(arm_values)) \
        if bandit_name.find("Exp") >= 0 else QuantileAdValueStrategy(np.max(arm_values), args.min_std_q)
    is_learn_visits = True if bandit_name[-1] == 'V' else False

    if bandit_name in ["JBExp", "JBExpV", "JBQV", "JBQ"]:
        bandit = JointBanditDiscriminatory(ads_learner=ads_bandit, price_learner=price_bandit_list, campaign=campaign,
                                           ad_value_strategy=ad_value_strategy, is_learn_visits=is_learn_visits)
    elif bandit_name in ["JBIExpV", "JBIQV"]:
        bandit = JointBanditDiscriminatoryImproved(ads_learner=ads_bandit, price_learner=price_bandit_list,
                                                   campaign=campaign,
                                                   ad_value_strategy=ad_value_strategy)
    elif bandit_name in ["JBBQ", "JBBExp"]:
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), args.init_std, args.alpha, args.n_restart_opt,
                                normalized=True) for _ in range(campaign.get_n_sub_campaigns())]
        bandit = JointBanditBalanced(campaign=campaign, arm_values=arm_values, price_learner_class=price_bandit_class,
                                     price_learner_kwargs=price_bandit_kwargs, number_of_visit_model_list=model_list,
                                     ad_value_strategy=ad_value_strategy)
    elif bandit_name == "JBFQ":
        assert args.daily_price, "This joint bandit requires to run in a daily manner"

        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), args.init_std, args.alpha, args.n_restart_opt,
                                normalized=True) for _ in range(campaign.get_n_sub_campaigns())]
        bandit = JointBanditFixedDailyPriceQuantile(campaign=campaign, number_of_visit_model_list=model_list,
                                                    min_std=args.min_std_q, arm_profit=arm_values,
                                                    n_arms_profit=len(arm_values))
    elif bandit_name == "JBFTS":
        assert args.daily_price, "This joint bandit requires to run in a daily manner"

        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), args.init_std, args.alpha, args.n_restart_opt,
                                normalized=True) for _ in range(campaign.get_n_sub_campaigns())]
        bandit = JointBanditFixedDailyPriceTS(campaign=campaign, number_of_visit_model_list=model_list,
                                              arm_profit=arm_values,
                                              n_arms_profit=len(arm_values))
    else:
        raise argparse.ArgumentParser("The name of the bandit to be used is not in the available ones")

    return bandit


def learn_per_user(bandit, env, campaign, prices, arm_profit):
    for _ in range(0, args.n_days):
        # Fix the price and simulate the day
        curr_budget_idx = bandit.pull_budget()
        curr_budget = [int(campaign.get_budgets()[i]) for i in curr_budget_idx]
        env.set_budget_allocation(budget_allocation=curr_budget)
        elapsed_day = not env.next_day()

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


def learn_per_day(bandit, env, campaign, prices, arm_profit, n_subcampaigns):
    explored_classes = np.zeros(n_subcampaigns)
    daily_price_idx = np.zeros(n_subcampaigns)

    for _ in range(0, args.n_days):
        # Fix budget
        # Fix the price and simulate the day
        curr_budget_idx = bandit.pull_budget()
        curr_budget = [int(campaign.get_budgets()[int(i)]) for i in curr_budget_idx]
        env.set_budget_allocation(budget_allocation=curr_budget)
        elapsed_day = not env.next_day()

        while not elapsed_day:
            # Retrieve user class
            user_class = env.next_user()[1]

            # Choose arm
            if not explored_classes[user_class]:
                daily_price_idx[user_class] = bandit.pull_price(user_class=user_class)
                explored_classes[user_class] = 1

            price_idx = int(daily_price_idx[user_class])

            # Observe reward
            reward, elapsed_day = env.round(price=prices[price_idx])
            reward = reward * arm_profit[price_idx]

            # Update bandit
            bandit.update_price(pulled_arm=price_idx, user_class=user_class, observed_reward=reward)

        # The day is over, update the budgeting model
        bandit.update_budget(pulled_arm_list=curr_budget_idx, n_visits=env.get_daily_visits_per_sub_campaign())
        explored_classes[:] = 0


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

    # choose whether to pull one arm a day for each class, or to pull one arm for each user
    if args.daily_price:
        learn_per_day(bandit, env, campaign, prices, arm_profit, scenario.get_n_subcampaigns())
    else:
        learn_per_user(bandit, env, campaign, prices, arm_profit)

    return bandit.get_daily_reward(), env.day_breakpoints


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
    total_rewards, day_breakpoint = main(args=args)
    print("Done run {}".format(id))
    return total_rewards, day_breakpoint


# Scheduling runs: ENTRY POINT
args = get_arguments()

seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
if args.n_jobs == 1:
    results = [run(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds)]
else:
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id=id, seed=seed, args=args) for id, seed in zip(range(args.n_runs), seeds))

total_rewards = [res[0] for res in results]
day_breakpoint = [res[1] for res in results]

if args.save_result:
    # Set up writing folder and file
    fd, folder_path_with_date = handle_folder_creation(result_path=args.output_folder)

    # Writing results and experiment details
    print("Storing results on file...", end="")
    with open("{}total_reward_{}.pkl".format(folder_path_with_date, args.joint_bandit_name), "wb") as output:
        pickle.dump(total_rewards, output)
    with open("{}day_{}.pkl".format(folder_path_with_date, args.joint_bandit_name), "wb") as output:
        pickle.dump(day_breakpoint, output)
    print("Done")

    fd.write("Bandit experiment\n")
    fd.write("Daily run mode: {}\n".format(args.daily_price))
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
    fd.write("Min quantile standard deviation (For quantile bandits): {}\n".format(args.min_std_q))
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
