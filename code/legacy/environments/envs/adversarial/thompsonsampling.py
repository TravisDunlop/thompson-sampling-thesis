import gym
from gym import error, spaces, utils
import numpy as np
from numpy.random import beta
import sys
sys.path.append('code')
import environments
from environments.envs.adversarial.abstract import Adversarial_Env

class Adversarial_ThompsonSampling(Adversarial_Env):
  ''' '''

  def get_name(self):
    return 'Adversarial-thompson-sampling'

  def sample_curr_step(self):
      self.truth[self.curr_step] = beta(self.truth_sucesses, self.truth_failures)
      self.expert_advice[:, self.curr_step] = beta(self.expert_successes, self.expert_failures)

  def reset(self, num_experts = 10, num_steps = 100):
      self.num_experts = num_experts
      self.num_steps = num_steps

      self.truth_sucesses = 1
      self.truth_failures = 1

      self.expert_successes = np.ones(num_experts)
      self.expert_failures = np.ones(num_experts)

      self.truth = np.zeros(num_steps)
      self.expert_advice = np.zeros((num_experts, num_steps))


      super().reset()

  def get_advice(self):
      self.sample_curr_step()
      return super().get_advice()
