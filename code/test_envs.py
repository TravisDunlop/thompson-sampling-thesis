import gym
import environments
import policies
import pandas as pd
import numpy as np

def test_policy(env, pol, num_episodes, results, reset_kwargs):
    '''Takes python gym style environment and tests
        given policy (pol) num_episodes amount of times
        then appends to results'''
    keys = sorted(reset_kwargs.keys())
    for episode in range(num_episodes):
        total_cost, cost = 0, 0
        env.reset(**reset_kwargs)
        pol.reset(env)
        done = False
        observation = None

        while not done:
            action = pol.act(observation)
            total_cost += cost

            observation, cost, done, info = env.step(action)
            pol.update(observation, cost)

        result = [env.get_name(), pol.get_name()]
        for key in keys: result.append(reset_kwargs[key])
        result.append(total_cost / num_steps)

        results.append(result)

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
