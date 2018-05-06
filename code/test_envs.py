import gym
import sys
sys.path.append('code')
import environments
import policies
from util import test_policy
import pandas as pd
import numpy as np

###
# TESTING



###
path = r'/Users/travisdunlop/Documents/thompson-sampling-thesis/'
num_experts = 10
environments = ['PWEA-iid-v0', 'PWEA-iid-w-switch-v0', 'PWEA-markov-v0']
pols = []
pols += [policies.policy.Random()]
pols += [policies.PWEA.ThompsonSampling()]
pols += [policies.PWEA.Exponential('PLG 2.2'), policies.PWEA.Exponential('PLG 2.3')]
pols += [policies.PWEA.Exponential('equation 13')]
pols += [policies.PWEA.FPL('uniform')]
pols += [policies.PWEA.FPL('exponential 2.2'), policies.PWEA.FPL('exponential 2.3')]
#pols += [policies.PWEA.FPL('random walk')]
#policies.PWEA.FPL('dropout')]
steps = np.random.randint(1, 1000, 1000)

results = []

for environment in environments:
    env = gym.make(environment)
    for pol in pols:
        for num_steps in steps:
            reset_kwargs = { 'num_experts' : num_experts, 'num_steps' : num_steps }
            test_policy(env, pol, 1, results, reset_kwargs)

results_df = pd.DataFrame(results, columns = ['environment', 'policy', 'experts', 'steps', 'cost_per_step'])

results_df.sample(5)

results_df.to_csv(path + 'data/results.csv', index = False)
