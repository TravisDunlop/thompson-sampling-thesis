import gym
import numpy as np
import sys
sys.path.append('code')
import environments
import policies
from util import test_policy
import pickle

import evolution
env = gym.make('PWEA-iid-v0')
pol = policies.FPL('dropout')

with open('data/adversarial evolutionary/populations.pickle', 'rb') as f:
    populations = pickle.load(f)

help(evolution)
