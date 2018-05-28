from ..policy import Policy
import numpy as np

class Exponential(Policy):
    '''Exponential Weights with worst case learning rates as described in
        Prediction, Learning, and Games by Cesa-Bianchi, Lugosi 2006 Sections
        2.2 and 2.3'''
    def __init__(self, learning_rate = 'worst-case'):
        self.learning_rate = learning_rate

    def reset(self, env):
        self.env = env

        self.loss = []
        self.weights = np.ones(self.env.num_experts)
        self.weights = self.weights / sum(self.weights)

    def get_name(self):
        return 'exponential weights - ' + self.learning_rate

    def get_lr(self):
        if self.learning_rate == 'PLG 2.2':
            return np.sqrt(np.log(self.env.num_experts) / self.env.num_steps)
        elif self.learning_rate == 'PLG 2.3':
            return np.sqrt(np.log(self.env.num_experts) / self.env.curr_step)
        elif self.learning_rate == 'equation 13':
            E = 1 # bound on rewards
            C = np.sqrt(2 * (np.sqrt(2) - 1) / (np.e - 2))
            V = max([1, np.var(self.loss)]) # variance of forecaster's rewards
            eta = min([1/E, C * np.sqrt(np.log(self.env.num_experts) / V)])
            return eta
        elif self.learning_rate == 'AdaHedge':
            return
        else:
            raise Exception('learning_rate parameter not recognized')

    def update(self, advice, observation):
        lr = self.get_lr()
        expert_regret = np.sum(self.env.get_expert_regret(), axis = 1)
        self.weights = np.exp( lr * expert_regret)
        self.weights = self.weights / sum(self.weights)

    def act(self, advice):
        best_expert = np.random.multinomial(1, self.weights).argmax()
        action = advice[best_expert]
        return action
