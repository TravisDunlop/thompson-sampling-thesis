import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('code')
import environments
from environments.envs.PWEA.abstract import PWEA_Env

class PWEA_iid(PWEA_Env):
  '''Prediction with Expert Advice, as described in Prediction Learning and
    Games (Cesa-Bianchi, Lugosi 2006).  Experts have some pre-specified bias
    and varying levels of gaussian noise.  Policy should learn which experts
    to weight heavier in making predictions '''

  def get_name(self):
    return 'PWEA-iid'

  def reset(self, num_experts = 10, num_steps = 100):
      self.num_experts = num_experts
      self.num_steps = num_steps

      self.truth = np.random.uniform(size = num_steps)
      self.expert_advice = np.random.uniform(size = (num_experts, num_steps))
      oracle = np.random.randint(num_experts)
      self.expert_advice[oracle] = self.truth

      super().reset()
