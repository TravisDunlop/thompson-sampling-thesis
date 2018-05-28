import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class Adversarial_Env(gym.Env):
  '''Prediction with Expert Advice'''
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.is_initalized = False
    self.action_space = spaces.Box(low = 0, high = 1, shape = (1,), dtype = np.float32)

  def reset(self):
      self.is_initalized = True
      self.curr_step = 0

      self.expert_loss = self.loss_function(self.expert_advice, self.truth)
      best_expert = self.expert_loss.min(axis = 1).argmin()
      self.best_expert_loss = self.expert_loss[best_expert, :]

      self.forecaster_prediction = np.zeros(self.num_steps)
      self.forecaster_loss = np.zeros(self.num_steps)
      self.forecaster_regret = np.zeros(self.num_steps)

  def step(self, action):
    if not self.is_initalized: raise Exception('environment not initialized: please call env.reset()')

    #observation
    observation = self.truth[self.curr_step]

    #loss
    loss = float(self.loss_function(observation, action))
    regret = loss - self.best_expert_loss[self.curr_step]

    self.forecaster_prediction[self.curr_step] = action
    self.forecaster_loss[self.curr_step] = loss
    self.forecaster_regret[self.curr_step] = regret

    #done
    self.curr_step += 1
    if self.curr_step == self.num_steps:
        done = True
    else:
        done = False

    #info
    info = ''
    return observation, loss, done, info

  def get_advice(self):
      return self.expert_advice[:, self.curr_step]

  def get_expert_regret(self):
      return self.forecaster_loss[:self.curr_step] - self.expert_loss[:, :self.curr_step]

  def get_expert_loss_prev(self):
      return self.expert_loss[:, self.curr_step - 1]

  def get_expert_loss_cummulative(self):
      return np.sum(self.expert_loss[:, :self.curr_step], axis = 1)

  def render(self, mode='human', close=False):
    ...

  def loss_function(self, x, y):
      return np.abs(x - y)

  def regret_per_step(self):
      return sum(self.forecaster_regret) / self.curr_step

class Adversarial_Env_No_Truth(gym.Env):
  '''Prediction with Expert Advice with only loss vectors'''
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.is_initalized = False
    self.action_space = spaces.Box(low = 0, high = 1, shape = (1,), dtype = np.float32)

  def reset(self):
      self.is_initalized = True
      self.curr_step = 0

      best_expert = self.expert_loss.min(axis = 1).argmin()
      self.best_expert_loss = self.expert_loss[best_expert, :]

      self.forecaster_prediction = np.zeros(self.num_steps, dtype = int)
      self.forecaster_loss = np.zeros(self.num_steps)
      self.forecaster_regret = np.zeros(self.num_steps)

  def step(self, action):
    if not self.is_initalized: raise Exception('environment not initialized: please call env.reset()')

    #observation
    observation = self.expert_loss[:, self.curr_step]

    #loss
    loss = self.expert_loss[action, self.curr_step]
    regret = loss - self.best_expert_loss[self.curr_step]

    self.forecaster_prediction[self.curr_step] = action
    self.forecaster_loss[self.curr_step] = loss
    self.forecaster_regret[self.curr_step] = regret

    #done
    self.curr_step += 1
    if self.curr_step == self.num_steps:
        done = True
    else:
        done = False

    #info
    info = ''
    return observation, loss, done, info

  def get_expert_regret(self):
      return self.forecaster_loss[:self.curr_step] - self.expert_loss[:, :self.curr_step]

  def get_expert_loss_prev(self):
      return self.expert_loss[:, self.curr_step - 1]

  def get_expert_loss_cummulative(self):
      return np.sum(self.expert_loss[:, :self.curr_step], axis = 1)

  def regret_per_step(self):
      return sum(self.forecaster_regret) / self.curr_step

  def loss_per_step(self):
      return sum(self.forecaster_loss) / self.curr_step
