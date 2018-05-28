import gym
from gym import error, spaces, utils
import numpy as np
from numpy.random import beta
import sys
sys.path.append('code')
import environments
from environments.envs.adversarial.abstract import Adversarial_Env_No_Truth

class Predetermined_No_Truth(Adversarial_Env_No_Truth):
  ''' '''

  def get_name(self):
    return 'Adversarial-predetermined-no-truth'

  def reset(self, expert_loss):
      self.num_experts, self.num_steps = expert_loss.shape

      self.expert_loss = expert_loss

      super().reset()
