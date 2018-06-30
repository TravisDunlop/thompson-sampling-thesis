import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('code')
import environments
from environments.envs.PWEA.abstract import PWEA_Env

class PWEA_iid_w_switch(PWEA_Env):
  '''Prediction with Expert Advice - one expert is always right, at one point
     which expert is right switches'''

  def get_name(self):
    return 'PWEA-iid-w-switch'

  def reset(self, num_experts = 10, num_steps = 100):
      self.num_experts = num_experts
      self.num_steps = num_steps

      self.truth = np.random.uniform(size = num_steps)
      self.expert_advice = np.random.uniform(size = (num_experts, num_steps))

      first_expert = np.random.randint(num_experts)
      second_expert = np.random.randint(num_experts)
      while first_expert == second_expert:
          second_expert = np.random.randint(num_experts)

      switch_step = int(num_steps / 2)

      self.expert_advice[first_expert, :switch_step] = self.truth[:switch_step]
      self.expert_advice[second_expert, switch_step:] = self.truth[switch_step:]

      super().reset()
