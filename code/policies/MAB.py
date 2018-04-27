
from .policy import Policy

import numpy as np

class ThompsonSampling(Policy):
    '''Thompson Sampling algorithm with Beta-prior
        for general bounded bandits. Described in
        Agrawal, Goyal 2012, Algorithm 2.'''
    def reset(self, env):
        super().reset(env)
        num_arms = env.action_space.n
        self.successes = [1] * num_arms #artifically start with 1 sucess and failure
        self.failures = [1] * num_arms

    def get_name(self):
        return 'thompson sampling'

    def update(self, action, reward):
        '''Perform Bernoulli trial, if success then update arm, otherwise not'''
        trial = np.random.binomial(1, reward)
        if trial == 1:
            self.successes[action] += 1
        else:
            self.failures[action] += 1

    def act(self):
        sample = np.random.beta(self.successes, self.failures)
        action = np.argmax(sample)
        return action
