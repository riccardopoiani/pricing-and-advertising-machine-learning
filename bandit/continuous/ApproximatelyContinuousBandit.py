import numpy as np

from bandit.IBandit import IBandit
from bandit.discrete.DiscreteBandit import DiscreteBandit
from utils.discretization.IteratedDiscretization import IteratedDiscretization


class ApproximatelyContinuousBandit(IBandit):

    def __init__(self, discretization: IteratedDiscretization, init_arm_values: np.array, bandit_class,
                 n_arms: int, **bandit_hp):
        super().__init__()
        self.discretization: IteratedDiscretization = discretization
        self.curr_arm_values: np.array = init_arm_values
        self.bandit: DiscreteBandit = bandit_class(n_arms=n_arms, **bandit_hp)
        self.bandit_hp = bandit_hp
        self.n_arms = n_arms
        self.bandit_class = bandit_class

    def pull_arm(self):
        self.bandit.pull_arm()

    def update(self, pulled_arm, reward):
        self.bandit.update(pulled_arm=pulled_arm, observed_reward=reward)

        if self.discretization.check_update_condition(time=self.bandit.t):
            # Reset the bandit, store information, and obtain new arms
            arm_history = [self.curr_arm_values[arm] for arm in self.bandit.pulled_arm_list]
            self.discretization.update_dataset(interval_reward_history=self.bandit.collected_rewards,
                                               interval_arm_history=arm_history)
            model = self.discretization.fit_model()
            self.curr_arm_values = self.discretization.discretize(model=model, n_arms=self.n_arms)
            self.bandit: DiscreteBandit = self.bandit_class(self.n_arms, **self.bandit_hp)

    def get_curr_arm_values(self):
        return self.curr_arm_values.copy()