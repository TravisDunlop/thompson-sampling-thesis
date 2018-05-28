from ..policy import Policy
import numpy as np

class FPL(Policy):
    '''Follow the Perturbed Leader policies'''
    def __init__(self, perturbation_type = 'exponential'):
        self.perturbation_type = perturbation_type

    def reset(self, env):
        self.env = env
        self.perturbation = np.zeros(self.env.num_experts)

    def get_name(self):
        return 'FPL - ' + self.perturbation_type

    def update(self, advice, observation):
        self.perturb()

    def perturb(self):
        if self.perturbation_type == 'uniform':
            '''Described in Prediction, Learning, and Games Exercise 4.7'''
            delta = np.sqrt(self.env.curr_step * self.env.num_experts)
            self.perturbation = np.random.uniform(0, delta, self.env.num_experts)
        elif self.perturbation_type == 'exponential 2.2':
            eta = np.sqrt(8 * np.log(self.env.num_experts) / self.env.num_steps) # assuming that the minimum loss is zero
            self.perturbation = np.random.laplace(scale = 1/eta, size = self.env.num_experts)
        elif self.perturbation_type == 'exponential 2.3':
            eta = np.sqrt(8 * np.log(self.env.num_experts) / self.env.curr_step) # assuming that the minimum loss is zero
            self.perturbation = np.random.laplace(scale = 1/eta, size = self.env.num_experts)
        elif self.perturbation_type == 'random walk':
            self.perturbation += np.random.choice([-0.5, 0.5], self.env.num_experts, [0.5, 0.5])
        elif self.perturbation_type == 'dropout':
            experts_to_drop = np.random.choice([0, 1], self.env.num_experts, [0.5, 0.5])
            self.perturbation -= experts_to_drop * self.env.get_expert_loss_prev()
        elif self.perturbation_type == 'beta':
            pass
        else:
            raise Exception('perturbation_type not recognized')

    def get_metric(self):
        '''get metric to perturb.  Usually this is cummulative loss, except
            when perturbation_type is beta then we perturb
        '''
        if self.perturbation_type == 'beta':
            pass
        else:
            return self.env.get_expert_loss_cummulative()

    def act(self, advice):
        metric = self.get_metric()
        best_expert = (metric + self.perturbation).argmin()
        action = advice[best_expert]
        return action

test = np.random.choice([0, 1], 5, [0.5, 0.5])

test
loss = np.random.uniform(size = 5)

test = test * loss
