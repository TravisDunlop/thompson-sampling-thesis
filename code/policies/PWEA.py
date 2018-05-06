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
        self.prev_advice = None
        self.step = 1
        self.costs = []

        self.weights = np.ones(self.num_experts)

    def get_name(self):
        return 'exponential weights - ' + self.learning_rate

    def loss_function(self, prediction, truth):
        return (truth - prediction) ** 2

    def get_lr(self):
        if self.learning_rate == 'PLG 2.2':
            return np.sqrt(np.log(self.num_experts) / self.num_steps)
        elif self.learning_rate == 'PLG 2.3':
            return np.sqrt(np.log(self.num_experts) / self.step)
        elif self.learning_rate == 'equation 13':
            E = 1 # bound on rewards
            C = np.sqrt(2 * (np.sqrt(2) - 1) / (np.e - 2))
            V = max([1, np.var(self.costs)]) # variance of forecaster's rewards
            eta = min([1/E, C * np.sqrt(np.log(self.num_experts) / V)])
            return eta
        elif self.learning_rate == 'AdaHedge':
            return
        else:
            raise Exception('learning_rate parameter not recognized')

    def update(self, observation, cost):
        if self.prev_advice is None: return

        prev_y = observation[0]
        loss = self.loss_function(self.prev_advice, prev_y)
        self.costs.append(cost)


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
    def __init__(self, perturbation_type = 'exponential'):
        self.perturbation_type = perturbation_type

    def reset(self, env):
        self.num_experts = env.num_experts
        self.num_steps = env.num_steps
        self.step = 1
        self.advice = None

        self.loss = np.zeros(self.num_experts)
        self.perturbation = np.zeros(self.num_experts)

    def get_name(self):
        return 'FPL - ' + self.perturbation_type

    def update(self, observation, cost):
        if observation is None: return

        prev_y = observation[0]
        if self.advice is not None:
            loss = self.loss_function(self.advice, prev_y)
        else:
            loss = np.zeros(self.num_experts)

        self.loss += loss
        self.perturb()

        self.step += 1
        self.advice = observation[1]

    def loss_function(self, prediction, truth):
        return (truth - prediction) ** 2

    def perturb(self):
        if self.perturbation_type == 'uniform':
            '''Described in Prediction, Learning, and Games Exercise 4.7'''
            delta = np.sqrt(self.step * self.num_experts)
            self.perturbation = np.random.uniform(0, delta, self.num_experts)
        elif self.perturbation_type == 'exponential 2.2':
            eta = np.sqrt(8 * np.log(self.num_experts) / self.num_steps) # assuming that the minimum loss is zero
            self.perturbation = np.random.laplace(scale = 1/eta, size = self.num_experts)
        elif self.perturbation_type == 'exponential 2.3':
            eta = np.sqrt(8 * np.log(self.num_experts) / self.step) # assuming that the minimum loss is zero
            self.perturbation = np.random.laplace(scale = 1/eta, size = self.num_experts)
        elif self.perturbation_type == 'random walk':
            self.perturbation += np.random.choice([-0.5, 0.5], self.num_experts, [0.5, 0.5])
        elif self.perturbation_type == 'dropout':
            pass
        else:
            raise Exception('perturbation_type not recognized')

    def act(self, observation = None):
        if observation is None: return 0
        best_expert = (self.loss + self.perturbation).argmin()
        action = self.advice[best_expert]
        return action

class ThompsonSampling(Policy):
    '''Thompson Sampling'''
    def __init__(self):
        pass

    def reset(self, env):
        self.num_experts = env.num_experts
        self.num_steps = env.num_steps
        self.step = 1
        self.advice = None

        self.successes = np.ones(self.num_experts)
        self.failures = np.ones(self.num_experts)

    def get_name(self):
        return 'thompson sampling'

    def update(self, advice, observation):
        if observation is None: return

        prev_y = observation[0]
        if self.advice is not None:
            loss = self.loss_function(self.advice, prev_y)
        else:
            loss = np.zeros(self.num_experts)

        trial = np.random.binomial(1, loss)

        self.successes += trial
        self.failures += 1 - trial

        self.step += 1
        self.advice = observation[1]

    def loss_function(self, prediction, truth):
        return (truth - prediction) ** 2

    def act(self, observation = None):
        if observation is None: return 0
        best_expert = np.random.beta(self.successes, self.failures).argmin()
        action = self.advice[best_expert]
        return action


np.var([1, 2, 4])
