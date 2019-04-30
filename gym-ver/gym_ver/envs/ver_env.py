import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import pdb


class network:
  def __init__(self, n):
    self.n = n

  def generate(self):
    raise NotImplementedError()

  
class uniform_network(network):
  def generate(self):
    samples = np.random.uniform(size = self.n)
    #samples = np.random.normal(size = self.n)
    #samples = np.vstack((np.zeros(self.n), samples))
    #samples = np.apply_along_axis(np.max, 0, samples)
    assert np.all(samples >= 0)
    return np.sum(samples)

class normal_network(network):
  def generate(self):
    samples = np.random.normal(size = self.n)
    samples = np.vstack((np.zeros(self.n), samples))
    samples = np.apply_along_axis(np.max, 0, samples)
    assert np.all(samples >= 0)
    return np.sum(samples)

  
class VerEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # These will be neural network generated
    self.ver = None
    self.load = uniform_network(3)

    self.capacity = 10
    self.cur_energy = 0
    self.cost_to_generate = 1
    self.cost_blackout = 1000

  def get_state(self):
    return (self.cur_energy, self.capacity)

  def set_costs(self, generation, blackout):
    self.cost_to_generate = generation
    self.cost_blackout = blackout
  
  def step(self, action):
    #pdb.set_trace()
    # get current load
    cur_load = self.load.generate()

    # Generate total cost
    # cost is price per created unit. Large drawback if load greater than
    # current energy, i.e., blackout
    cost = self.cost_to_generate * action
    if cur_load >= self.cur_energy + action:
      cost += self.cost_blackout

    # update state
    self.cur_energy += action
    self.cur_energy -= cur_load
    self.cur_energy = np.maximum(self.cur_energy, 0)
    if self.cur_energy < 0:
      pdb.set_trace()
    
    # return obs, reward, done, info
    return self.get_state(), -cost, False, None
    
  def reset(self):
    self.cur_energy = 0
    return self.get_state()
    
  def render(self, mode='human', close=False):
    print('rEnDeRiNg:', self.cur_energy)