import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class PWEA_markov(gym.Env):
  '''All experts have a chance of being right, which expert it is switches
     according to a Markov Process initialized randomly at the beginning'''
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.num_experts, self.num_steps, self.current_step = None, None, None
    self.is_inialized = False

  def get_name(self):
    return 'PWEA-markov'

  def step(self, action):
    if not self.is_inialized: raise Exception('environment not initialized: please call env.reset()')
    #state
    self.prev_y = self.y
    self.y = np.random.uniform()

    #observation
    self.update_expert() # take a step in markov chain
    curr_advice = np.random.uniform(size = self.num_experts)
    curr_advice[self.curr_expert] = self.y
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

  def update_expert(self):
      self.curr_expert = np.random.multinomial(1, self.transition_probs[self.curr_expert]).argmax()

  def reset(self, num_experts = 10, num_steps = 100):
      self.is_inialized = True
      self.current_step = 0
      self.num_experts = num_experts
      self.num_steps = num_steps

      #drawing transition probabilities from dirichlet distribution
      self.transition_probs = np.zeros((self.num_experts, self.num_experts))
      for i in range(self.num_experts):
          mean = np.ones(self.num_experts)
          mean[i] = 100
          self.transition_probs[i] = np.random.dirichlet(mean)

      self.curr_expert = np.random.randint(self.num_experts)

      self.y, self.prev_y = 0, 0

      self.action_space = spaces.Box(low = 0, high = 1, shape = (1,), dtype = np.float32)

  def render(self, mode='human', close=False):
    ...

  def cost_function(self, x, y):
      return (x - y) ** 2
