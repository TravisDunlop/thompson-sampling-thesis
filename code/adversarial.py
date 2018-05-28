
import gym
import sys
sys.path.append('code')
import environments
import policies
import pickle

import numpy as np

def array2string(array):
    if isinstance(array, list): array = np.array(array)
    return np.array2string(array, precision = 3, separator = ',').replace('\n', '')

class Individual:
    def __init__(self, expert_loss):
        self.expert_loss = expert_loss
        self.num_experts, self.num_steps = expert_loss.shape
        self.regret_per_step = []
        self.loss_per_step = []

    def test_fitness(self, env, pol):
        env.reset(self.expert_loss)
        pol.reset(env)

        done = False

        while not done:
            action = pol.act()
            observation, cost, done, info = env.step(action)
            pol.update(observation)

        self.regret_per_step.append(round(env.regret_per_step(), 3))
        self.loss_per_step.append(round(env.loss_per_step()))

    def make_child(self, mutation_var):
        expert_loss = self.expert_loss.copy()
        mutation = np.random.normal(size = expert_loss.shape, scale = mutation_var)
        expert_loss += mutation
        expert_loss = expert_loss.clip(0, 1)
        child = Individual(expert_loss)
        return child

    def mean_regret_per_step(self):
        return np.mean(self.regret_per_step)

    def mean_loss_per_step(self):
        return np.mean(self.loss_per_step)

    def num_trials(self):
        return len(self.regret_per_step)

    def write(self, f, type = 'long'):
        line = [str(self.num_experts), str(self.num_steps), str(self.mean_regret_per_step())]
        line.extend([str(self.mean_loss_per_step()), str(self.num_trials())])
        if type == 'short':
            pass
        elif type == 'long':
            line.append(array2string(self.expert_loss))
            line.append(array2string(self.regret_per_step))
            line.append(array2string(self.loss_per_step))
        else:
            raise('Type not recognized')
        return f.write('|'.join(line) + '\n')

class Population:
    def __init__(self, num_experts, num_steps, population_size, mutation_var, percent_to_kill):
        self.num_experts = num_experts
        self.num_steps = num_steps
        self.population_size = population_size
        self.mutation_var = mutation_var
        self.percent_to_kill = percent_to_kill
        self.num_generations = 0

        expert_losses = [np.random.uniform(size = (num_experts, num_steps)) for _ in range(population_size)]
        self.members = [Individual(expert_loss) for expert_loss in expert_losses]


    def generation(self, env, pol):
        if self.num_generations == 0: self.test_fitness(env, pol)
        self.sort_and_kill()
        self.make_children()
        self.test_fitness(env, pol)
        self.num_generations += 1

    def test_fitness(self, env, pol):
        for individual in self.members:
            individual.test_fitness(env, pol)

    def sort_and_kill(self):
        #sort on mean_regret_per_step
        mean_regret_per_step = np.array([i.mean_regret_per_step() for i in self.members])
        sort_index = np.argsort(-mean_regret_per_step).tolist()
        self.members = [self.members[i] for i in sort_index]

        #kill percent_to_kill members
        cutoff = round(len(self.members) * (1 - percent_to_kill))
        self.members, to_kill = self.members[:cutoff], self.members[cutoff:]

    def write(self, f_short, f_long):
        for individual in self.members:
            individual.write(f_short, 'short')
            individual.write(f_long, 'long')

    def make_children(self):
        new_generation = [individual.make_child(self.mutation_var) for individual in self.members]
        self.members.extend(new_generation)


population_size = 100
num_generations = 100
num_experts_range = range(2, 50, 5)
num_steps_range = range(10, 250, 20)
mutation_var = 0.05
percent_to_kill = 0.5
short_col = ['num_experts', 'num_steps', 'mean_regret_per_step', 'mean_loss_per_step', 'num_trials']
long_col = short_col + ['expert_loss', 'regret_per_step', 'loss_per_step']

kwargs = {'population_size' : population_size, \
          'mutation_var' : mutation_var,
          'percent_to_kill' : percent_to_kill}

env = gym.make('Adversarial-predetermined-no-truth-v0')
pol = policies.ThompsonSampling_No_Truth()

def save():
    with open('data/adversarial evolutionary/populations.pickle', 'wb') as f:
        pickle.dump(populations, f)

    with open('data/adversarial evolutionary/short_data.txt', 'w') as f_short:
        f_short.write('|'.join(short_col) + '\n')

    with open('data/adversarial evolutionary/long_data.txt', 'w') as f_long:
        f_long.write('|'.join(long_col) + '\n')

    with open('data/adversarial evolutionary/short_data.txt', 'a') as f_short:
        with open('data/adversarial evolutionary/long_data.txt', 'a') as f_long:
            for population in populations:
                population.write(f_short, f_long)

#initialize population
#populations = [Population(e,s, **kwargs) for e in num_experts_range for s in num_steps_range]

for generation in range(num_generations):
    for population in populations:
        population.generation(env, pol)
    save()
    print('Finished Generation: {}'.format(generation))
