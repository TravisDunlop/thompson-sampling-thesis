import gym
import multi_armed_bandit
from multi_armed_bandit import policy
import pandas as pd

def test_policy(env, pol, num_episodes, results):
    '''Takes python gym style environment and tests
        given policy (pol) num_episodes amount of times
        then appends results as list of lists in results.'''
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

        results.append(['random', num_arms, num_steps, total_reward])


num_arms, num_steps, num_episodes = 5, 100, 3

env = gym.make('MAB-iid-v0')
pol = policy.Random()

results = []

test_policy(env, pol, num_episodes, results)

results
