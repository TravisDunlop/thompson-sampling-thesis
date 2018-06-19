'''In this file we make a function which iterates through all
   possibilities of loss matricies for a given number of actions
   and time steps and reports the maximum expected regret'''

from itertools import product
from collections import Counter
from time import time

import numpy as np
from numpy import argmax, dot
from numpy.random import beta, randint

############################################################
# helper functions

def mean(lst): return sum(lst) / len(lst)

def get_loss_matricies(action_space, num_timesteps, num_actions):
    '''generate a list of tuples with all possible combinations of the num_options
        for a length num_items.  This is used to initialize all starting values'''
    raw_tuples = list(product(action_space, repeat = num_timesteps * num_actions))
    loss_matrixs = [np.array(t).reshape((num_timesteps, num_actions)) for t in raw_tuples]
    return loss_matrixs

def summarize_actions(actions, num_actions):
    '''take a vector of actions and create a list showing the percentage of times
        the action was taken.  In this case, an action is a value between 0 and
        num_levers - 1 corresponding to the lever chosen.'''
    num_estimations = len(actions)
    counter = Counter(actions)
    action_pct = [counter[i] / num_estimations for i in range(num_actions)]
    return action_pct

def estimate_forecaster_loss(loss, cum_loss, step, num_actions, num_estimations):
    '''estimates instantaneous forecaster loss for the given time step'''
    a = [1 + i for i in cum_loss]
    b = [1 + step - i for i in cum_loss]

    # sampling num_estimations actions
    actions = [beta(a, b).argmin() for _ in range(num_estimations)]
    action_pct = summarize_actions(actions, num_actions)

    forecaster_loss = dot(loss, action_pct)

    return forecaster_loss

def estimate_regret(loss_matrix, num_estimations):
    '''this function estimates the expected cummulative regret of a loss matrix'''
    num_timesteps, num_actions = loss_matrix.shape

    best_action = loss_matrix.sum(axis = 0).argmin()
    best_action_loss = loss_matrix[:, best_action].sum()

    cum_loss = np.zeros(num_actions)
    forecaster_loss = 0

    for step, loss in enumerate(loss_matrix):
        cum_loss += loss
        forecaster_loss += estimate_forecaster_loss(loss, cum_loss, step + 1, num_actions, num_estimations)

    expected_cumulative_regret = forecaster_loss - best_action_loss
    return expected_cumulative_regret

def expected_max_regret(num_actions, num_timesteps, action_space, num_estimations):
    loss_matricies = get_loss_matricies(action_space, num_timesteps, num_actions)

    regrets = np.array([estimate_regret(loss_matrix, num_estimations) for loss_matrix in loss_matricies])

    order = (-regrets).argsort()

    max_regret = mean([regrets[i] for i in order[:10]])
    top_loss_matricies = [loss_matricies[i] for i in order[:10]]

    return max_regret, top_loss_matricies

############################################################

num_actions = 2
num_timesteps = 5
action_space = [0, 1]
num_estimations = 100

current_time = time()
results = []
for a in range(2, 11):
    for t in range(5, 21, 5):
        #expected_max_regret, top_loss_matricies = 0, 0
        max_regret, top_loss_matricies = expected_max_regret(a, t, action_space, num_estimations)
        results.append((a, t, expected_max_regret, top_loss_matricies))
        print('num_actions: {} | num_timesteps: {} | {}'.format(a, t, time() - current_time))

results
