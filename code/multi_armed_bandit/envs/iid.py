import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class MAB_iid(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.arms, self.num_steps, self.current_step = None, None, None
    self.is_inialized = False

  def step(self, action):
    if not self.is_inialized: raise Exception('environment not initialized: please call env.reset()')
    #observation
    observation = ''

    #reward
    a, b = self.arms[action]
    reward = np.random.beta(a, b)

    #done
    self.current_step += 1
    if self.current_step == self.num_steps:
        done = True
    else:
        done = False

    #info
    info = ''
    return observation, reward, done, info
  def reset(self, num_arms = 5, num_steps = 100):
      self.is_inialized = True
      self.current_step = 0
      self.arms = np.random.uniform(size = (num_arms, 2))
      self.num_steps = num_steps

      self.action_space = spaces.Discrete(num_arms)

  def render(self, mode='human', close=False):
    ...
