import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class PWEA_iid(gym.Env):
  '''Prediction with Expert Advice, as described in Prediction Learning and
    Games (Cesa-Bianchi, Lugosi 2006).  Experts have some pre-specified bias
    and varying levels of gaussian noise.  Policy should learn which experts
    to weight heavier in making predictions '''
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.num_experts, self.num_steps, self.current_step = None, None, None
    self.is_inialized = False

  def get_name(self):
    return 'PWEA-iid'

  def step(self, action):
    if not self.is_inialized: raise Exception('environment not initialized: please call env.reset()')
    #state
    self.prev_y = self.y
    self.y = np.random.uniform()

    #observation
    curr_advice = np.random.uniform(size = self.num_experts)
    curr_advice[0] = self.y
    observation = (self.prev_y, curr_advice)

    #reward
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
      self.is_inialized = True
      self.current_step = 0
      self.num_experts = num_experts
      self.num_steps = num_steps

      #! self.a = np.random.uniform(size = num_experts)

      self.y, self.prev_y = 0, 0

      self.action_space = spaces.Box(low = 0, high = 1, shape = (1,), dtype = np.float32)

  def render(self, mode='human', close=False):
    ...

  def cost_function(self, x, y):
      return (x - y) ** 2
