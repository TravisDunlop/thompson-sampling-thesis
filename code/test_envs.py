import gym
import sys
sys.path.append('code')
import environments
import policies
from util import test_policy, test_and_save, save_list
import pandas as pd
import numpy as np

###
# TESTING



###
folder = r'/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis/data/'
num_experts = 10
min_step = 10
max_step = 1000
num_simulations = 1000

environments = ['PWEA-iid-v0', 'PWEA-iid-w-switch-v0']
environments += ['PWEA-markov-v0']
pols = []
#pols += [policies.policy.Random()]
#pols += [policies.ThompsonSampling()]
pols += [policies.Exponential('PLG 2.2'), policies.Exponential('PLG 2.3')]
#pols += [policies.Exponential('equation 13')]
#pols += [policies.Exponential('AdaHedge')]
#pols += [policies.FPL('uniform')]
#pols += [policies.FPL('exponential 2.2'), policies.FPL('exponential 2.3')]
#pols += [policies.FPL('random walk')]
#pols += [policies.FPL('dropout')]

for environment in environments:
    env = gym.make(environment)
    for pol in pols:
        test_and_save(env, pol, folder, num_experts, min_step, max_step, num_simulations)

#testing markov with different dirichlet_factor
num_steps = 200
num_experts = 10
min_dirichlet_factor = 1 / num_experts
max_dirichlet_factor = 100
dirichlet_factors = steps = np.random.uniform(min_dirichlet_factor, max_dirichlet_factor, 1000)

env = gym.make('PWEA-markov-v0')
results = []

for pol in pols:
    for dirichlet_factor in dirichlet_factors:
        reset_kwargs = { 'num_experts' : num_experts, 'num_steps' : num_steps, 'dirichlet_factor' : dirichlet_factor }
        test_policy(env, pol, 1, results, reset_kwargs)


file_name = folder + 'testing dirichlet factors/results.csv'
save_list(results, file_name)
