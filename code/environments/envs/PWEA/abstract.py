import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class PWEA_Env(gym.Env):
  '''Prediction with Expert Advice'''
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.is_initalized = False
    self.action_space = spaces.Box(low = 0, high = 1, shape = (1,), dtype = np.float32)

  def step(self):
    if not self.is_initalized: raise Exception('environment not initialized: please call env.reset()')

  def render(self, mode='human', close=False):
    ...

  def cost_function(self, x, y):
      return np.abs(x - y)

  def regret_per_step(self):
      return sum(self.forecaster_regret) / self.num_steps
