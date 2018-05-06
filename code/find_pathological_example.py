''' This script attempts to identify pathological examples where Thompson
    Sampling performs terribly. I focus on cases where there are two experts
    in the 'prediction with expert advice' and full observation of the outcome.'''

import gym
import numpy as np
import sys
sys.path.append('code')
import environments
import policies
from util import test_policy

population_size = 100
num_generations = 100
num_steps = 10
mutation_var = 0.01

#initialize population
env = gym.make('PWEA-predetermined-v0')
pol = policies.PWEA.ThompsonSampling()
population = {'advice' = [], 'y' = [], 'costs' = [], 'mean_cost' = []}

for generation in range(num_generations):
    #test fitness
    fitness_population(env, pol, population)

    #kick out bottom 1/2

    #recombine top 1/4

    #perturb top 1/4


def fitness_individual(env, pol, population, individual):
    advice = population['advice'][individual]
    y = population['y'][individual]
    reset_kwargs = {'advice' : advice, 'y' : y}
    results = []
    test_policy(env, pol, 2, results, reset_kwargs)
    losses = [result[4] for result in results]
    loss = np.mean(losses)

    population['loss'][individual] = loss

def fitness_population(env, pol, population):

    #test fitness - update loss estimates
    population_size = len(population['advice'])
    for individual in range(population_size):
        test_individual(env, pol, population, individual)

    #reorder population by decreasing fitness



expert_advice = np.array((np.random.uniform(size = num_steps), np.random.uniform(size = num_steps)))
truth = np.random.uniform(size = num_steps)

advice.shape

expert_advice

truth

expert_cost = env.cost_function(expert_advice, truth)
best_expert = expert_cost.min(axis = 1).argmin()

best_expert_cost = expert_cost[best_expert, :]

expert_regret = expert_cost - best_expert_cost


reset_kwargs = {'advice' : advice, 'y' : y}

s = [1, 6, 4, 5]


np.mean([1, 0])

results = []
test_policy(env, pol, 1, results, reset_kwargs)

_, _, advice, y, loss = results[0]


total_cost, cost = 0, 0
env.reset(**reset_kwargs)
pol.reset(env)
done = False
observation = None

###
while not done:
    action = pol.act(observation)
    total_cost += cost

    observation, cost, done, info = env.step(action)
    pol.update(observation, cost)

result = [env.get_name(), pol.get_name()]
for key in keys: result.append(reset_kwargs[key])
result.append(total_cost / num_steps)
####

help(env.step)
