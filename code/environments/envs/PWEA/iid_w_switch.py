import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class PWEA_iid_w_switch(gym.Env):
  '''Prediction with Expert Advice - one expert is always right, at one point
     which expert is right switches'''
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.is_initalized = False

  def get_name(self):
    return 'PWEA-iid-w-switch'

  def step(self, action):
    if not self.is_initalized: raise Exception('environment not initialized: please call env.reset()')
    #state
    self.prev_y = self.y
    self.y = np.random.uniform()

    #observation
    curr_advice = np.random.uniform(size = self.num_experts)
    if self.current_step < self.switch_step:
        curr_advice[self.first_expert] = self.y
    else:
        curr_advice[self.second_expert] = self.y
    observation = (self.prev_y, curr_advice)

    #cost
    cost = float(self.cost_function(self.prev_y, action))

    #done
    self.current_step += 1
    if self.current_step == self.num_steps:
        done = True
    else:
        done = False

    #info
    info = ''
    return observation, cost, done, info

  def reset(self, num_experts = 10, num_steps = 100):
      self.is_initalized = True
      self.current_step = 0
      self.num_experts = num_experts
      self.num_steps = num_steps

      self.first_expert = np.random.randint(num_experts)
      self.second_expert = np.random.randint(num_experts)

      self.switch_step = int(num_steps / 2)

      self.y, self.prev_y = 0, 0

      self.action_space = spaces.Box(low = 0, high = 1, shape = (1,), dtype = np.float32)

  def render(self, mode='human', close=False):
    ...

  def cost_function(self, x, y):
      return (x - y) ** 2
