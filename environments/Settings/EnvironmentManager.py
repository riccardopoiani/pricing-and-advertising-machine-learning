from typing import List
import os, json
import numpy as np

from environments.Phase import Phase
from utils.folder_management import get_resource_folder_path
from utils.stats.StochasticFunction import IStochasticFunction, BoundedLambdaStochasticFunction


class EnvironmentManager(object):

    @classmethod
    def create_n_clicks_function(cls, function_dict: dict) -> IStochasticFunction:
        """
        Create the number of clicks function given a dictionary of template:
            {
                "type": ...
                "info": {
                    ...
                }
            }

        :param function_dict: dictionary containing info related to the function
        :return: an IStochasticFunction representing the number of clicks chosen
        """
        if function_dict["type"] == "linear":
            def linear_generator_function(coefficient, bias, lower_bound, upper_bound, noise_std):
                if lower_bound is None:
                    lower_bound = -np.inf
                if upper_bound is None:
                    upper_bound = np.inf

                return lambda x: \
                    max(int(np.random.normal(np.maximum(np.minimum(coefficient * x + bias, upper_bound), lower_bound),
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
    def create_crp_function(cls, function_dict: dict) -> IStochasticFunction:
        """
        Create the conversion rate probability function given a dictionary of template:
            {
                "type": ...
                "info": {
                    ...
                }
            }

        :param function_dict: the dictionary about the conversion rate probability to create
        :return: an IStochasticFunction representing the conversion rate probability
        """
        if function_dict["type"] == "linear":
            def linear_generator_function(coefficient, min_price, max_crp):
                return lambda x: \
                    np.random.binomial(n=1, p=np.max([0, coefficient * (-x + min_price) + max_crp]))

            function_info = function_dict["info"]
            fun: IStochasticFunction = BoundedLambdaStochasticFunction(
                f=linear_generator_function(function_info["coefficient"],
                                            function_info["min_price"],
                                            function_info["max_crp"]),
                min_value=function_info["min_price"], max_value=function_info["max_price"])
        else:
            raise NotImplementedError("Type of function not available")
        return fun

    @classmethod
    def load_scenario(cls, scenario_name, verbose=False) -> (List[Phase]):
        """
        Load a scenario on the basis of its name

        :param verbose: whether to print out information regarding the environment
        :param scenario_name: name of the scenario which is also the filename of the json file saved in the resources
                              folder
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
            phases: List[Phase] = []
            for i, phase_dict in enumerate(data_phases):
                if verbose:
                    print("EnvironmentManager:\t {} - {}".format(i + 1, phase_dict["phase_name"]))

                crp_functions: List[IStochasticFunction] = []
                n_clicks_functions: List[IStochasticFunction] = []

                for crp_function in phase_dict["crp_functions"]:
                    crp_functions.append(cls.create_crp_function(crp_function))

                for n_clicks_function in phase_dict["n_clicks_functions"]:
                    n_clicks_functions.append(cls.create_n_clicks_function(n_clicks_function))

                # Verify that the crp functions and n_clicks functions has both "n_subcampaigns" functions
                assert len(crp_functions) == data["n_subcampaigns"]
                phases.append(Phase(phase_dict["duration"], n_clicks_functions, crp_functions))
            return phases
