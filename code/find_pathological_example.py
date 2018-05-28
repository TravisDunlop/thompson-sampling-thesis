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

population_size = 200
num_generations = 10000
num_experts = 2
num_steps = 10
mutation_var = 1
percent_to_cut = 0.5

def make_children(population, mutation_var):
    num_children = len(population['expert_advice'])
    population['expert_advice'] += [mutate(a, mutation_var) for a in population['expert_advice']]
    population['truth'] += [mutate(a, mutation_var) for a in population['truth']]
    population['regret_per_step'] += [[] for _ in range(num_children)]

def mutate(array, mutation_var):
    array = array.copy()
    mutation = np.random.normal(size = array.shape, scale = mutation_var)
    array += mutation
    array = array.clip(0, 1)
    array = (2 * array).round() / 2
    return array

def sort_population(population, percent_to_cut = 0.5):
    cutoff = round(len(population['expert_advice']) * percent_to_cut)
    avg_regret_per_step = np.array([np.mean(r) for r in population['regret_per_step']])
    sort_index = np.argsort(-avg_regret_per_step).tolist()

    for key in population.keys():
        population[key] = [population[key][i] for i in sort_index[:cutoff]]

def fitness_individual(env, pol, population, individual):
    advice = population['expert_advice'][individual]
    y = population['truth'][individual]
    reset_kwargs = {'expert_advice' : advice, 'truth' : y}
    results = []
    test_policy(env, pol, 2, results, reset_kwargs)
    regret_per_step = [result[4] for result in results]
    population['regret_per_step'][individual] += regret_per_step

def fitness_population(env, pol, population):

    #test fitness - update loss estimates
    population_size = len(population['expert_advice'])
    for individual in range(population_size):
        fitness_individual(env, pol, population, individual)

#initialize population
env = gym.make('PWEA-predetermined-v0')
pol = policies.ThompsonSampling()

expert_advice = [np.random.uniform(size = (num_experts, num_steps)) for _ in range(population_size)]
truth = [np.random.uniform(size = num_steps) for _ in range(population_size)]
regret_per_step = [[] for _ in range(population_size)]

population = {'expert_advice' : expert_advice, 'truth' : truth, 'regret_per_step' : regret_per_step}

for generation in range(num_generations):
    #test fitness
    fitness_population(env, pol, population)

    #kick out bottom 1/2
    sort_population(population, percent_to_cut)

    #perturb top 1/2
    make_children(population, mutation_var)

population['expert_advice'][0]
population['truth'][0]
population['regret_per_step'][1]
