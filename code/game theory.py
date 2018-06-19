
import numpy as np
import pandas as pd
from numpy.random import beta, binomial
%matplotlib inline

from numpy import log

import os
os.chdir('/Users/travis/Documents/Education/Barcelona GSE/thesis/thompson-sampling-thesis')

class ThompsonSamplingAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.reset()

    def reset(self):
        self.a = np.ones(num_actions)
        self.b = np.ones(num_actions)

    def act(self):
        return beta(self.a, self.b).argmin()

    def update(self, loss_vector):
        trial = binomial(1, loss_vector)

        self.a += trial
        self.b += 1 - trial

def calculate_regret(game_record):
    num_turns = len(game_record)
    agent_1_loss = sum([loss for action_1, action_2, loss in game_record])
    agent_2_loss = sum([1 - loss for action_1, action_2, loss in game_record])

    agent_1_actions = [0 for _ in range(num_actions)]
    agent_2_actions = [0 for _ in range(num_actions)]

    for action_1, action_2, loss in game_record:
        agent_1_actions[action_1] += 1
        agent_2_actions[action_2] += 1

    best_fixed_action_loss_1 = min(agent_2_actions)
    best_fixed_action_loss_2 = min(agent_1_actions)

    agent_1_regret = agent_1_loss - best_fixed_action_loss_1
    agent_2_regret = agent_2_loss - best_fixed_action_loss_2

    return agent_1_regret, agent_2_regret


def play_game(agent_1, agent_2, game_matrix, num_turns):
    agent_1.reset()
    agent_2.reset()

    game_record = []

    for step in range(num_turns):
        action_1 = agent_1.act()
        action_2 = agent_2.act()

        loss = game_matrix[action_1, action_2]

        agent_1.update(game_matrix[:, action_2])
        agent_2.update(1 - game_matrix[action_1, :])

        game_record.append((action_1, action_2, loss))

    agent_1_regret, agent_2_regret = calculate_regret(game_record)

    return game_record, agent_1_regret, agent_2_regret

num_actions = 5
num_turns = 500

game_matrix = np.identity(num_actions)

actions_range = range(2, 30)
turns_range = range(10, 200, 10)
iterations = range(100)

results = []

for num_actions in actions_range:
    print('now trying num_actions: {}'.format(num_actions))
    game_matrix = np.identity(num_actions)

    agent_1 = ThompsonSamplingAgent(num_actions)
    agent_2 = ThompsonSamplingAgent(num_actions)

    for num_turns in turns_range:
        for i in iterations:
            game_record, agent_1_regret, agent_2_regret = play_game(agent_1, agent_2, game_matrix, num_turns)
            results.append((num_actions, num_turns, agent_1_regret, agent_2_regret))

results_df = pd.DataFrame(results, columns = ['num_actions', 'num_turns', 'agent_1_regret', 'agent_2_regret'])

results_df.to_csv('data/game_theory_results.csv', index = False)

results_df = pd.read_csv('data/game_theory_results.csv')

results_df.head()

ax = results_df.plot('num_turns', 'agent_1_regret', 'scatter', logx = True, logy = True, ylim = (1, 130));

ax.plot(turns_range, turns_range)

results_df.plot('num_actions', 'agent_1_regret', 'scatter');
