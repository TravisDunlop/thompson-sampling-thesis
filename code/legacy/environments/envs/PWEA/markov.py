import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import sys
sys.path.append('code')
import environments
from environments.envs.PWEA.abstract import PWEA_Env

class PWEA_markov(PWEA_Env):
  '''All experts have a chance of being right, which expert it is switches
     according to a Markov Process initialized uniformly randomly at the beginning'''

  def get_name(self):
    return 'PWEA-markov'

  def reset(self, num_experts = 10, num_steps = 100, dirichlet_factor = 10):
      self.num_experts = num_experts
      self.num_steps = num_steps

      self.truth = np.random.uniform(size = num_steps)
      self.expert_advice = np.random.uniform(size = (num_experts, num_steps))

      #drawing transition probabilities from dirichlet distribution
      transition_probs = np.zeros((num_experts, num_experts))
      for i in range(num_experts):
          mean = np.ones(num_experts)
          mean[i] = num_experts * dirichlet_factor
          transition_probs[i] = np.random.dirichlet(mean)

      #oracles are indices of which experts to follow
      oracles = np.zeros(10, dtype = int)
      oracles[0] = np.random.randint(num_experts)

      #follow the markov chain for next expert
      for i in range(1, len(oracles)):
          oracles[i] = np.random.multinomial(1, transition_probs[oracles[i-1]]).argmax()

      #overwriting expert_advice depending on which expert is the oracle
      for i, oracle in enumerate(oracles):
          self.expert_advice[oracle, i] = self.truth[i]

      super().reset()
