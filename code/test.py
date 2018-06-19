import gym
import numpy as np
import sys
sys.path.append('code')
import environments
import policies
from util import test_policy
import pickle

from evolution import Population, Individual

env = gym.make('Adversarial-predetermined-no-truth-v0')
pol = policies.ThompsonSampling_No_Truth()

with open('data/adversarial evolutionary/populations.pickle', 'rb') as f:
    populations = pickle.load(f)

for individual in populations[0].members:
    print(individual.expert_loss)
    break


expert_advice = np.array(([1/2] + [1, 0] * 5, [1/2] + [0, 1] * 5))

env.reset(expert_advice)

i = Individual(expert_advice)
i.test_fitness(env, pol)

i.mean_regret_per_step()
 #------------------------------------------------

 types = [(population.num_steps, population.num_experts) for population in populations]

types[:20]

population = populations[0]

for individual in population.members:
    for _ in range(100): individual.test_fitness(env, pol)

mean_regret_per_step = [i.mean_regret_per_step() for i in population.members]

np.argmax(mean_regret_per_step)

# ------------------------------------------------

i = populations[0].members[94]

i.mean_regret_per_step()

len(i.regret_per_step)

test_i = Individual((i.expert_loss * 2).round() / 2)

test_i.expert_loss

for _ in range(100): test_i.test_fitness(env, pol)

test_i.mean_regret_per_step()
