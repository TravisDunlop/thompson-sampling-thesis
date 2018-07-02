
from agents import ThompsonSamplingAgent, FPLDropout, ExponentialWeightedAverage

import numpy as np
import pandas as pd
%matplotlib inline
import seaborn as sb

from numpy import log

def play_game(agent_1, agent_2, payoff_matrix, num_turns):
    num_actions, _ = payoff_matrix.shape

    agent_1.reset(num_actions)
    agent_2.reset(num_actions)

    #agent 1 loss
    agent_1_loss = 0
    total_loss_1 = np.zeros(agent_1.num_actions)
    total_loss_2 = np.zeros(agent_2.num_actions)

    for step in range(num_turns):
        action_1 = agent_1.act()
        action_2 = agent_2.act()

        loss = payoff_matrix[action_1, action_2]
        loss_vector_1 = payoff_matrix[:, action_2]
        loss_vector_2 = 1 - payoff_matrix[action_1, :]

        agent_1.update(loss_vector_1)
        agent_2.update(loss_vector_2)

        agent_1_loss += loss
        total_loss_1 += loss_vector_1
        total_loss_2 += loss_vector_2

    agent_2_loss = num_turns - agent_1_loss
    agent_1_regret = agent_1_loss - min(total_loss_1)
    agent_2_regret = agent_2_loss - min(total_loss_2)

    return agent_1_loss, agent_2_loss, agent_1_regret, agent_2_regret

############################################################

num_actions = 5
num_turns = 10

payoff_matrix = np.identity(num_actions)

agent_1 = ThompsonSamplingAgent()
agent_2 = ThompsonSamplingAgent()

agent_1.reset(num_actions)
agent_2.reset(num_actions)

play_game(agent_1, agent_2, payoff_matrix, num_turns)

############################################################
#generating payoff matrices

def identity(num_actions):
    return np.identity(num_actions)

def rock_paper_scissors(num_actions):
    payoff_matrix = 1/2 * np.identity(num_actions)
    for i in range(1 - num_actions, num_actions):
        #set every other diagonal to ones, odd diagonals for lower triangular
        # even diagonals for upper triangular
        if ( i < 0 and i % 2 == 1 ) or ( i > 0 and i % 2 == 0 ):
            payoff_matrix += np.diag(np.ones(num_actions - abs(i)), i)

    return payoff_matrix

def uniform(num_actions):
    return np.random.uniform(size = (num_actions, num_actions))

############################################################
agents_1 = [ThompsonSamplingAgent(), FPLDropout(), ExponentialWeightedAverage()]
agents_2 = [ThompsonSamplingAgent(), FPLDropout(), ExponentialWeightedAverage()]
payoffs = [identity, rock_paper_scissors, uniform]
#actions_range = range(2, 30)
actions_range = [2, 15, 30]
turns_range = np.logspace(1, 3, num = 15).astype('int')
iterations = range(100)

results = []
for num_actions in actions_range:
    print('now trying num_actions: {}'.format(num_actions))

    for payoff in payoffs:
        payoff_matrix = payoff(num_actions)
        for agent_1 in agents_1:
            for agent_2 in agents_2:

                for num_turns in turns_range:
                    for i in iterations:
                        result = play_game(agent_1, agent_2, payoff_matrix, num_turns)
                        results.append((num_turns, num_actions, payoff.__name__, \
                                        agent_1.type(), agent_2.type()) + result)

cols = ['num_turns', 'num_actions', 'payoff_type', 'agent_1', 'agent_2',
        'loss_1', 'loss_2', 'regret_1', 'regret_2']
results_df = pd.DataFrame(results, columns = cols)

# path = '../../data/game_theory_results_.csv'

path = '../data/game_theory_results_dueling.csv'

results_df.to_csv(path, index = False)

results_df = pd.read_csv(path)

results_df.head()
