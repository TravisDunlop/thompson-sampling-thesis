import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('code')
import environments
from environments.envs.PWEA.abstract import PWEA_Env

class PWEA_predetermined(PWEA_Env):
  '''Prediction with Expert Advice - loss vectors set by user'''

  def __init__(self):
    super()

  def get_name(self):
    return 'PWEA-predetermined'

  def reset(self, expert_advice, truth):
      self.is_initalized = True
      self.curr_step = 0
      self.num_experts = expert_advice.shape[0]
      self.num_steps = truth.shape[0]

      self.truth = truth
      self.expert_advice = expert_advice

      self.expert_cost = self.cost_function(expert_advice, truth)
      best_expert = self.expert_cost.min(axis = 1).argmin()
      self.best_expert_cost = self.expert_cost[best_expert, :]
      self.expert_regret = self.expert_cost - self.best_expert_cost

      self.forecaster_prediction = np.zeros(self.num_steps)
      self.forecaster_cost = np.zeros(self.num_steps)
      self.forecaster_regret = np.zeros(self.num_steps)

  def get_advice(self):
      return self.expert_advice[:, self.curr_step]

  def step(self, action):
    super()

    #observation
    y = self.truth[self.curr_step]
    observation = y

    #cost
    cost = float(self.cost_function(y, action))
    regret = cost - self.best_expert_cost[self.curr_step]

    self.forecaster_prediction[self.curr_step] = action
    self.forecaster_cost[self.curr_step] = cost
    self.forecaster_regret[self.curr_step] = regret

    #done
    self.curr_step += 1
    if self.curr_step == self.num_steps:
        done = True
    else:
        done = False

    #info
    info = ''
    return observation, cost, done, info
