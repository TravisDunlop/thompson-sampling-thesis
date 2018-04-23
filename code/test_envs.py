import gym
import multi_armed_bandit
from multi_armed_bandit import policy
import pandas as pd

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

        results.append([env.get_name(), pol.get_name(), num_arms, num_steps, total_reward])


path = r'/Users/travisdunlop/Documents/thompson-sampling-thesis/'
num_arms, num_episodes = 5, 100

env = gym.make('MAB-iid-v0')
pol = policy.Random()

results = []

for num_steps in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    test_policy(env, pol, num_steps, num_episodes, results)

results_df = pd.DataFrame(results, columns = ['environment', 'policy', 'num_arms', 'num_steps', 'total_reward'])

results_df.head()

results_df.to_csv(path + 'code/data/results.csv', index = False)
