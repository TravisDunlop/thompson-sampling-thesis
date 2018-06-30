from ..policy import Policy
import numpy as np

class ThompsonSampling(Policy):
    '''Thompson Sampling'''
    def __init__(self):
        pass

    def reset(self, env):
        self.env = env

        self.successes = np.ones(self.env.num_experts)
        self.failures = np.ones(self.env.num_experts)

    def get_name(self):
        return 'thompson sampling'

    def update(self, advice, observation):
        y = observation
        loss = self.env.loss_function(advice, y)

        trial = np.random.binomial(1, loss)

        self.successes += trial
        self.failures += 1 - trial

    def act(self, advice):
        best_expert = np.random.beta(self.successes, self.failures).argmin()
        action = advice[best_expert]
        return action

class ThompsonSampling_No_Truth(Policy):
    '''Thompson Sampling'''
    def __init__(self):
        pass

    def reset(self, env):
        self.env = env

        self.successes = np.ones(self.env.num_experts)
        self.failures = np.ones(self.env.num_experts)

    def get_name(self):
        return 'thompson sampling - no truth'

    def update(self, observation):
        loss = observation

        trial = np.random.binomial(1, loss)

        self.successes += trial
        self.failures += 1 - trial

    def act(self):
        best_expert = np.random.beta(self.successes, self.failures).argmin()
        return best_expert
