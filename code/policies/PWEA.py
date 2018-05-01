from .policy import Policy
import numpy as np

class Exponential(Policy):
    '''Exponential Weights with worst case learning rates as described in
        Prediction, Learning, and Games by Cesa-Bianchi, Lugosi 2006 Sections
        2.2 and 2.3'''
    def __init__(self, learning_rate = 'worst-case'):
        self.learning_rate = learning_rate

    def reset(self, env):
        self.num_experts = env.num_experts
        self.num_steps = env.num_steps
        self.step = 1
        self.weights = np.ones(self.num_experts)
        self.prev_advice = None

    def get_name(self):
        return 'exponential weights - ' + self.learning_rate

    def loss_function(self, prediction, truth):
        return (truth - prediction) ** 2

    def get_lr(self):
        if self.learning_rate == 'PLG_2_2':
            return np.sqrt(8 * np.log(self.num_experts) / self.num_steps)
        elif self.learning_rate == 'PLG_2_3':
            return np.sqrt(8 * np.log(self.num_experts) / self.step)
        elif self.learning_rate == '':
            return
        else:
            raise Exception('learning_rate parameter not recognized')

    def update(self, observation):
        if self.prev_advice is None: return

        prev_y = observation[0]
        loss = self.loss_function(self.prev_advice, prev_y)
        lr = self.get_lr()
        exp_loss = np.exp( - lr * loss)
        self.weights = self.weights * exp_loss
        self.weights = self.weights / sum(self.weights)

        self.step += 1

    def act(self, observation = None):
        if observation is None:
            return 0
        else:
            advice = observation[1]
            self.prev_advice = advice
            return float(self.weights.dot(advice))

class FPL(Policy):
    '''Follow the Perturbed Leader policies'''
    def __init__(self, perturbation_type = ''):
        pass

    def reset(self):
        pass

    def get_name(self):
        pass

    def update(self, observation):
        pass

    def act(self, observation = None):
        pass

class ThompsonSampling(Policy):
    '''Thompson Sampling'''
    def __init__(self, perturbation_type = ''):
        pass

    def reset(self):
        pass

    def get_name(self):
        pass

    def update(self, observation):
        pass

    def act(self, observation = None):
        pass
