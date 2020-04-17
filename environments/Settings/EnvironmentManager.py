import json
import os
from typing import List, Dict, Tuple

import numpy as np

from environments.Settings.Phase import Phase
from environments.Settings.Scenario import Scenario
from utils.folder_management import get_resource_folder_path
from utils.stats.StochasticFunction import IStochasticFunction, BoundedLambdaStochasticFunction


class EnvironmentManager(object):

    @classmethod
    def create_n_clicks_function(cls, function_dict: dict, get_mean_function) -> IStochasticFunction:
        """
        Create the number of clicks function given a dictionary of template:
            {
                "type": ...
                "info": {
                    ...
                }
            }

        :param function_dict: dictionary containing info related to the function
        :param get_mean_function: whether to get the stochastic function or the mean function
        :return: an IStochasticFunction representing the number of clicks chosen
        """
        if function_dict["type"] == "linear":
            def linear_generator_function(coefficient, bias, lower_bound, upper_bound, noise_std):
                if lower_bound is None:
                    lower_bound = -np.inf
                if upper_bound is None:
                    upper_bound = np.inf

                if get_mean_function:
                    return lambda x: \
                        0 if x == 0 else max(int(np.maximum(np.minimum(coefficient * x + bias, upper_bound),
                                                            lower_bound)), 0)  # function returns positive int numbers
                else:
                    return lambda x: \
                        0 if x == 0 else max(int(
                            np.random.normal(np.maximum(np.minimum(coefficient * x + bias, upper_bound), lower_bound),
                                             noise_std)), 0)  # function returns positive int numbers

            function_info = function_dict["info"]
            fun: IStochasticFunction = BoundedLambdaStochasticFunction(
                linear_generator_function(function_info["coefficient"],
                                          function_info["bias"],
                                          function_info["lower_bound"],
                                          function_info["upper_bound"],
                                          function_info["noise_std"]))
        else:
            raise NotImplementedError("Type of function not available")
        return fun

    @classmethod
    def create_crp_function(cls, function_dict: dict, get_mean_function) -> IStochasticFunction:
        """
        Create the conversion rate probability function given a dictionary of template:
            {
                "type": ...
                "info": {
                    ...
                }
            }

        :param function_dict: the dictionary about the conversion rate probability to create
        :param get_mean_function: whether to get the mean function or the stochastic version
        :return: an IStochasticFunction representing the conversion rate probability
        """
        if function_dict["type"] == "linear":
            def linear_generator_function(coefficient, min_price, max_crp):
                if get_mean_function:
                    return lambda x: np.max([0, coefficient * (-x + min_price) + max_crp])
                else:
                    return lambda x: np.random.binomial(n=1, p=np.max([0, coefficient * (-x + min_price) + max_crp]))

            function_info = function_dict["info"]
            fun: IStochasticFunction = BoundedLambdaStochasticFunction(
                f=linear_generator_function(function_info["coefficient"],
                                            function_info["min_price"],
                                            function_info["max_crp"]),
                min_value=function_info["min_price"], max_value=function_info["max_price"])
        elif function_dict["type"] == "tanh":
            def tanh_generator_function(coefficient, x_offset, dilation, y_offset):
                if get_mean_function:
                    return lambda x: np.max([0, coefficient *
                                             np.tanh(x_offset - x / dilation) + y_offset])
                else:
                    return lambda x: np.random.binomial(n=1, p=np.max(
                        [0, coefficient * np.tanh(x_offset - x / dilation) + y_offset]))

            function_info = function_dict["info"]
            fun: IStochasticFunction = BoundedLambdaStochasticFunction(
                f=tanh_generator_function(coefficient=function_info["coefficient"],
                                          x_offset=function_info["x_offset"],
                                          dilation=function_info["dilation"],
                                          y_offset=function_info["y_offset"]),
                min_value=function_info["min_price"], max_value=function_info["max_price"]
            )
        else:
            print("here")
            raise NotImplementedError("Type of function not available: {}\n".format(function_dict["type"]))
        return fun

    @classmethod
    def get_crps_for_prices(cls, scenario_name: str, prices: List[float]) -> List[List[float]]:
        scenario_folder = get_resource_folder_path()
        scenario_json_file_path = os.path.join(scenario_folder, scenario_name + ".json")

        def get_crps_for_params(coefficient, min_price, max_crp, price: float):
            return np.max([0, coefficient * (-price + min_price) + max_crp])

        with open(scenario_json_file_path) as json_file:
            data: dict = json.load(json_file)
            data_phases: List = data["phases"]
            crps_per_phase: List[List[float]] = []
            for i, phase_dict in enumerate(data_phases):
                crps: List[float] = []
                for j, crp_dict in enumerate(phase_dict["crp_functions"]):
                    crp_dict = crp_dict["info"]
                    price = prices[j]
                    p = get_crps_for_params(crp_dict["coefficient"], crp_dict["min_price"], crp_dict["max_crp"], price)
                    crps.append(p)
                crps_per_phase.append(crps)
            return crps_per_phase

    @classmethod
    def load_scenario(cls, scenario_name: str, verbose=False, get_mean_function=False) -> Scenario:
        """
        Load a scenario on the basis of its name

        :param verbose: whether to print out information regarding the environment
        :param scenario_name: name of the scenario which is also the filename of the json file saved in the resources
                              folder
        :param get_mean_function: whether to get a stochastic function or a mean function
        :return: a list of phases contained in the scenario chosen
        """
        scenario_folder = get_resource_folder_path()
        scenario_json_file_path = os.path.join(scenario_folder, scenario_name + ".json")
        with open(scenario_json_file_path) as json_file:
            data: dict = json.load(json_file)

            if verbose:
                print("EnvironmentManager: the scenario imported is {}".format(data["scenario_name"]))
                print("EnvironmentManager: the scenario's number of subcampaigns is {}".format(data["n_subcampaigns"]))
                print("EnvironmentManager: the scenario's phases are")

            data_phases: List = data["phases"]
            n_subcampaigns: int = data["n_subcampaigns"]
            n_user_features: int = data["n_user_features"]
            phases: List[Phase] = []
            for i, phase_dict in enumerate(data_phases):
                if verbose:
                    print("EnvironmentManager:\t {} - {}".format(i + 1, phase_dict["phase_name"]))

                crp_functions: List[IStochasticFunction] = []
                n_clicks_functions: List[IStochasticFunction] = []

                for crp_function in phase_dict["crp_functions"]:
                    crp_functions.append(cls.create_crp_function(crp_function, get_mean_function=get_mean_function))

                for n_clicks_function in phase_dict["n_clicks_functions"]:
                    n_clicks_functions.append(cls.create_n_clicks_function(n_clicks_function,
                                                                           get_mean_function=get_mean_function))

                # Verify that the crp functions and n_clicks functions has both "n_subcampaigns" functions
                assert len(crp_functions) == n_subcampaigns
                phases.append(Phase(phase_dict["duration"], n_clicks_functions, crp_functions))

            min_context_to_subcampaign: Dict[Tuple[int], int] = {}
            for key, value in data["min_context_to_subcampaign"].items():
                user_tuple_key = tuple([int(value) for value in key.replace("(", "").replace(")", "").split(",")])

                assert n_subcampaigns > value >= 0  # verify that the min context refers to a valid subcampaign
                min_context_to_subcampaign[user_tuple_key] = value

            scenario = Scenario(n_subcampaigns, n_user_features, min_context_to_subcampaign, phases)
            return scenario
