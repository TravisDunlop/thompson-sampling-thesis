import gym
import numpy as np
import sys
sys.path.append('code')
import environments
import policies
from util import test_policy

env = gym.make('PWEA-predetermined-v0')
pol = policies.PWEA.ThompsonSampling()

num_steps = 10
expert_advice = np.array((np.random.uniform(size = num_steps), np.random.uniform(size = num_steps)))
truth = np.random.uniform(size = num_steps)

reset_kwargs = {'expert_advice' : expert_advice, 'truth' : truth}

cost = 0, 0
env.reset(**reset_kwargs)
pol.reset(env)
done = False
observation = None
action = 'initial'

while not done:
    advice = pol.act(advice)
    observation, cost, done, info = env.step(action)
    pol.update(advice, observation)

result = [env.get_name(), pol.get_name()]
for key in keys: result.append(reset_kwargs[key])
result.append(total_cost / env.num_steps)

results.append(result)

env.regret_per_step()

env.forecaster_regret
env.forecaster_cost
env.best_expert_cost
env.truth
env.forecaster_prediction
