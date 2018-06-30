
from agents import ExponentialWeightedAverage, FPLDropout

import numpy as np
from numpy.random import beta, binomial, uniform, choice
from numpy import where

num_actions = 5



losses = binomial(1, 0.5, size = num_actions)

choice(np.where(losses == losses.min())[0])

agent = FPLDropout(5)

agent.update(losses)

losses

agent.act()
