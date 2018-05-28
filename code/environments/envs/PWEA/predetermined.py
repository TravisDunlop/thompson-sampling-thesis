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

  def get_name(self):
    return 'PWEA-predetermined'

  def reset(self, expert_advice, truth):
      self.num_experts = expert_advice.shape[0]
      self.num_steps = truth.shape[0]

      self.expert_advice = expert_advice
      self.truth = truth

      super().reset()
