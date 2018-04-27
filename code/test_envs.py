import gym
import environments
import policies
import pandas as pd
import numpy as np

def test_policy(env, pol, num_steps, num_episodes, results):
    '''Takes python gym style environment and tests
        given policy (pol) num_episodes amount of times
        then appends to results'''
    for episode in range(num_episodes):
        total_reward, reward = 0, 0
        env.reset(num_arms, num_steps)
        pol.reset(env)
        done = False

        while not done:
            action = pol.act()
            total_reward += reward

            observation, reward, done, info = env.step(action)
            pol.update(action, reward)

        results.append([env.get_name(), pol.get_name(), num_arms, num_steps, total_reward / num_steps])

###
# TESTING


###
path = r'/Users/travisdunlop/Documents/thompson-sampling-thesis/'
num_arms, num_episodes = 5, 100

environments = ['MAB-iid-v0']
policies = [policies.policy.Random(), policies.MAB.ThompsonSampling()]
steps = np.random.randint(1, 1000, 1000)

results = []

for environment in environments:
    env = gym.make(environment)
    for pol in policies:
        for num_steps in steps:
            test_policy(env, pol, num_steps, 1, results)

results_df = pd.DataFrame(results, columns = ['environment', 'policy', 'arms', 'steps', 'reward_per_step'])

results_df.sample(5)

results_df.to_csv(path + 'data/results.csv', index = False)
