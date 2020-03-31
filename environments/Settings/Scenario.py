from typing import List, Tuple, Dict

from environments.Settings.Phase import Phase


class Scenario(object):
    """
    It represents a scenario on which the experiments is tested (simulation of the reality with a sort of model)

    Assumptions:
     - the number of subcampaigns remains constant through all the phases
     - the number of features can be different than the number of classes (==number of subcampaigns)
     - the number of phases of pricing is equal to the number of phases of advertisement
     - the time horizon of the pricing and advertisement has to be the same (checked during the generation of phases)
     - the sum of user distribution of all type of users (i.e. users with all features specified) is 1
    """

    def __init__(self, n_subcampaigns: int, n_user_features: int, user_distributions: Dict[Tuple[int], float],
                 phases: List[Phase]):
        # Assertions
        for phase in phases:
            assert n_subcampaigns == phase.get_n_subcampaigns()

        total_distribution = 0
        n_combination_user_features = 0
        for user_features, distribution in user_distributions.items():
            total_distribution += distribution
            n_combination_user_features += 1
        assert n_combination_user_features == n_user_features**2
        assert total_distribution == 1.0

        self.n_subcampaigns = n_subcampaigns
        self.n_user_features = n_user_features
        self.phases = phases
        self.user_distributions = user_distributions

    def get_n_subcampaigns(self):
        return self.n_subcampaigns

    def get_n_user_features(self):
        return self.n_user_features

    def get_phases(self):
        return self.phases

    def get_user_distribution(self):
        return self.user_distributions
