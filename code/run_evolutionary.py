
import pickle
import numpy as np
from numpy import mean
from numpy.random import uniform, normal, binomial
from agents import ThompsonSamplingAgent, FPLDropout, ExponentialWeightedAverage

def estimate_regret(loss_matrix, agent):
    num_actions, num_turns = loss_matrix.shape

    agent.reset(num_actions)
    agent_loss = 0

    for step in range(num_turns):
        action = agent.act()
        agent_loss += loss_matrix[action, step]
        agent.update(loss_matrix[:, step])

    best_fixed_action = loss_matrix.sum(axis = 1).min()

    regret = agent_loss - best_fixed_action
    return regret

class Individual:
    def __init__(self, loss_matrix, agent):
        self.loss_matrix = loss_matrix
        self.agent = agent
        self.regret_estimates = []

    def test_fitness(self, times = 2):
        for _ in range(times):
            regret_estimate = estimate_regret(self.loss_matrix, self.agent)
            self.regret_estimates.append(regret_estimate)

    def fitness(self):
        return mean(self.regret_estimates)

    def make_child(self, dropout = 0):
        noise = normal(scale = noise_sd, size = self.loss_matrix.shape)
        drop = binomial(1, dropout, size = self.loss_matrix.shape)
        noise *= drop

        loss_matrix = self.loss_matrix.copy() + noise
        loss_matrix = loss_matrix.clip(0, 1)

        return Individual(loss_matrix, self.agent)


    def make_children(self):
        children = [self.make_child(dropout) for dropout in dropouts]
        return children

class Population:
    def __init__(self, num_individuals, num_actions, num_turns, agent):
        # uniformly randomly initialize population
        self.num_actions = num_actions
        self.num_turns = num_turns
        self.agent = agent

        self.curr = 0

        self.members = []
        for _ in range(num_individuals):
            loss_matrix = uniform(size = (num_actions, num_turns))
            individual = Individual(loss_matrix, agent)
            self.members.append(individual)

    def generation(self):
        if self.curr == 0: self.test_fitness()
        self.sort_and_kill()
        self.make_children()
        self.test_fitness()
        self.curr += 1

    def test_fitness(self):
        for individual in self.members: individual.test_fitness()

    def sort_and_kill(self):
        #sort on mean_regret_per_step
        fitness = np.array([i.fitness() for i in self.members])
        sort_index = np.argsort(-fitness).tolist()
        self.members = [self.members[i] for i in sort_index]

        #kill percent_to_kill members
        cutoff = round(len(self.members) * (1 - percent_to_kill))
        self.members, to_kill = self.members[:cutoff], self.members[cutoff:]

    def write(self, f_short, f_long):
        for individual in self.members:
            individual.write(f_short, 'short')
            individual.write(f_long, 'long')

    def make_children(self):
        new_generation = []
        for individual in self.members:
            new_generation.extend(individual.make_children())

        self.members.extend(new_generation)

def save(populations, path = '../data/evolutionary/populations.pickle'):
    with open(path, 'wb') as f:
        pickle.dump(populations, f)

def load(path = '../data/evolutionary/populations.pickle'):
    with open(path, 'rb') as f:
        populations = pickle.load(f)
    return populations

#other hyperparameters
num_individuals = 100
noise_sd = 0.05
dropouts = [1, 0.25]
percent_to_kill = 2/3

agents = [ThompsonSamplingAgent(), FPLDropout(), ExponentialWeightedAverage()]
actions_range = range(2, 30, 4)
turns_range = np.logspace(1, 2, num = 10).astype('int')

# populations = []
# for num_actions in actions_range:
#     for agent in agents:
#         for num_turns in turns_range:
#             population = Population(num_individuals, num_actions, num_turns, agent)
#             populations.append(population)

populations = load()


while True:
    print('generation complete: {}'.format(populations[0].curr))
    for population in populations: population.generation()

    save(populations)

####
#Test
