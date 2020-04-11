import argparse
from typing import List

import numpy as np

from advertising.data_structure.Campaign import Campaign
from advertising.regressors.DiscreteGPRegressor import DiscreteGPRegressor
from advertising.regressors.DiscreteGaussianRegressor import DiscreteGaussianRegressor
from advertising.regressors.DiscreteRegressor import DiscreteRegressor
from bandit.combinatiorial.CDCombinatorialBandit import CDCombinatorialBandit
from bandit.combinatiorial.CombinatorialStationaryBandit import CombinatorialStationaryBandit
from bandit.combinatiorial.SWCombinatorialBandit import SWCombinatorialBandit
from bandit.context.ContextualBandit import ContextualBandit
from bandit.discrete.EXP3Bandit import EXP3Bandit
from bandit.discrete.TSBanditRescaledBernoulli import TSBanditRescaledBernoulli
from bandit.discrete.UCB1Bandit import UCB1Bandit
from bandit.discrete.UCB1MBandit import UCB1MBandit
from bandit.discrete.UCBLBandit import UCBLBandit
from bandit.discrete.UCBLM import UCBLMBandit


def get_bandit_class_and_kwargs(bandit_name: str, n_arms: int, arm_values: np.ndarray, args):
    if bandit_name == "TS":
        bandit_class = TSBanditRescaledBernoulli
        bandit_kwargs = {"n_arms": n_arms, "arm_values": arm_values}
    elif bandit_name == "UCB1":
        bandit_class = UCB1Bandit
        bandit_kwargs = {"n_arms": n_arms, "arm_values": arm_values}
    elif bandit_name == "UCB1M":
        bandit_class = UCB1MBandit
        bandit_kwargs = {"n_arms": n_arms, "arm_values": arm_values}
    elif bandit_name == "UCBL":
        bandit_class = UCBLBandit
        bandit_kwargs = {"n_arms": n_arms, "crp_upper_bound": args.crp_upper_bound, "arm_values": arm_values}
    elif bandit_name == "UCBLM":
        bandit_class = UCBLMBandit
        bandit_kwargs = {"n_arms": n_arms, "crp_upper_bound": args.crp_upper_bound, "arm_values": arm_values}
    elif bandit_name == "EXP3":
        bandit_class = EXP3Bandit
        bandit_kwargs = {"n_arms": n_arms, "gamma": args.gamma, "arm_values": arm_values}
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit_class, bandit_kwargs


def build_discrete_bandit(bandit_name: str, n_arms: int, arm_values: np.ndarray, args):
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param bandit_name:
    :param n_arms:
    :param arm_values: values of each arm
    :param args: command line arguments
    :return: bandit that will be used to carry out the experiment
    """
    if bandit_name == "TS":
        bandit = TSBanditRescaledBernoulli(n_arms=n_arms, arm_values=arm_values)
    elif bandit_name == "UCB1":
        bandit = UCB1Bandit(n_arms=n_arms, arm_values=arm_values)
    elif bandit_name == "UCB1M":
        bandit = UCB1MBandit(n_arms=n_arms, arm_values=arm_values)
    elif bandit_name == "UCBL":
        bandit = UCBLBandit(n_arms=n_arms, crp_upper_bound=args.crp_upper_bound, arm_values=arm_values)
    elif bandit_name == "UCBLM":
        bandit = UCBLMBandit(n_arms=n_arms, crp_upper_bound=args.crp_upper_bound, arm_values=arm_values)
    elif bandit_name == "EXP3":
        bandit = EXP3Bandit(n_arms=n_arms, gamma=args.gamma, arm_values=arm_values)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def build_combinatorial_bandit(bandit_name: str, campaign: Campaign, init_std, args):
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param bandit_name: the name of the bandit for the experiment
    :param campaign: the campaign over which the bandit needs to optimize the budget allocation
    :param init_std: initial standard deviation of the regressors
    :param args: arguments from arguments parser
    :return: bandit that will be used to carry out the experiment
    """
    if bandit_name == "GPBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), init_std, args.alpha, args.n_restart_opt,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CombinatorialStationaryBandit(campaign=campaign, model_list=model_list)
    elif bandit_name == "GaussianBandit":
        model_list: List[DiscreteRegressor] = [DiscreteGaussianRegressor(list(campaign.get_budgets()),
                                                                         init_std)
                                               for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CombinatorialStationaryBandit(campaign=campaign, model_list=model_list)
    elif bandit_name == "GPSWBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), init_std, args.alpha, args.n_restart_opt,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = SWCombinatorialBandit(campaign=campaign, model_list=model_list, sw_size=args.sw_size)
    elif bandit_name == "CDBandit":
        model_list: List[DiscreteRegressor] = [
            DiscreteGPRegressor(list(campaign.get_budgets()), init_std, args.alpha, args.n_restart_opt,
                                normalized=True)
            for _ in range(campaign.get_n_sub_campaigns())]
        bandit = CDCombinatorialBandit(campaign=campaign, model_list=model_list, n_arms=args.n_arms_ads,
                                       gamma=args.gamma, cd_threshold=args.cd_threshold, sw_size=args.sw_size)
    else:
        raise argparse.ArgumentError("The name of the bandit to be used is not in the available ones")

    return bandit


def build_contextual_bandit(bandit_name, n_arms, arm_values, n_features, context_generation_frequency, args):
    """
    Retrieve the bandit to be used in the experiment according to the bandit name

    :param bandit_name:
    :param n_arms:
    :param arm_values: values of each arm
    :param n_features: number of user features
    :param context_generation_frequency:
    :param args: command line arguments
    :return: bandit that will be used to carry out the experiment
    """
    bandit_class, bandit_kwargs = get_bandit_class_and_kwargs(bandit_name, n_arms, arm_values, args)
    bandit = ContextualBandit(n_features, args.confidence, context_generation_frequency, bandit_class,
                              **bandit_kwargs)
    return bandit
