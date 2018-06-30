
import numpy as np
from numpy import exp, sqrt, log, where
from numpy.random import beta, binomial, choice

class ThompsonSamplingAgent:
    def type(self):
        return 'Thompson Sampling'

    def reset(self, num_actions):
        self.num_actions = num_actions
        self.a = np.ones(self.num_actions)
        self.b = np.ones(self.num_actions)

    def act(self):
        return beta(self.a, self.b).argmin()

    def update(self, loss_vector):
        trial = binomial(1, loss_vector)

        self.a += trial
        self.b += 1 - trial

class FPLDropout:
    def type(self):
        return 'FPL Dropout'

    def reset(self, num_actions, dropout = 0.5):
        self.num_actions = num_actions
        self.dropout = dropout
        self.perturbed_loss = np.zeros(self.num_actions)

    def act(self):
        return choice(where(self.perturbed_loss == self.perturbed_loss.min())[0])

    def update(self, loss_vector):
        self.perturbed_loss += binomial(1, (1 - self.dropout) * loss_vector)

class ExponentialWeightedAverage:
    def type(self):
        return 'EWA'

    def reset(self, num_actions):
        self.num_actions = num_actions
        self.step = 0
        self.weights = np.ones(self.num_actions) / self.num_actions
        self.total_loss = np.zeros(self.num_actions)

    def act(self):
        return choice(self.num_actions, p = self.weights)

    def update(self, loss_vector):
        self.step += 1
        #use learning rate from Prediction, Learning, and Games 2.3
        lr = sqrt(log(self.num_actions) / self.step)
        self.total_loss += loss_vector
        self.weights = exp( -lr * self.total_loss)
        self.weights = self.weights / sum(self.weights)
